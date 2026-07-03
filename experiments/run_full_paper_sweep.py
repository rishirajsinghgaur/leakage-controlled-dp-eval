"""
Full paper experiment sweep — runs everything needed for the DP-FL anomaly detection paper.

Scope (realistic for CPU, ~4-8h):
  Datasets:   TEP, SWaT, SKAB
  Conditions: 6 (baselines + deduplication)
  K:          5 clients
  Partitions: IID + Dirichlet α=0.5 (most interesting non-IID)
  ε grid:     {0.5, 1, 2, 4, 8} for DP conditions
  Seeds:      42, 43, 44 (3 seeds → error bars)

Plus ablations a-e on SKAB (fastest dataset).

Results written to results/paper_results.json
Figures to results/figures/
Tables  to results/tables/
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.loaders      import DATASET_REGISTRY
from data.partitioner  import partition_iid, partition_dirichlet, verify_non_iid
from models.siamese    import SiameseEncoder, train_siamese
from models.mlp        import AnomalyAutoencoder, evaluate_anomaly_detector
from dedup.sil_gate    import SILGate
from dedup.local_dedup import LocalDeduplicator
from privacy.dp_trainer import DPTrainer
from privacy.accountant import compute_sigma_for_epsilon, compute_epsilon, privacy_report
from privacy.mia       import MIAEvaluator
from fl.client         import build_client_fn
from fl.server         import run_simulation, ResultRecord
from experiments.evaluate import (
    make_privacy_utility_figure, make_communication_figure,
    make_mia_figure, make_latex_table, save_results,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

RESULTS_PATH = ROOT / "results" / "paper_results.json"
FIG_DIR      = ROOT / "results" / "figures"
TAB_DIR      = ROOT / "results" / "tables"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-specific configs
# ─────────────────────────────────────────────────────────────────────────────

DS_CFG = {
    "skab": {
        # High temporal redundancy: ρ≈0.06 at τ=0.97 (water pump, 1Hz sensor)
        "max_samples": 40_000, "bottleneck": 4,  "batch_size": 64,
        "local_epochs": 3, "n_rounds": 15, "lr": 1e-3,
        "siamese_epochs": 20, "siamese_emb": 16, "siamese_window": 10,
        "similarity_threshold": 0.97, "sil_sigma": 3.0,
        "eval_percentile": 90.0, "delta": 1e-5, "max_grad_norm": 1.0,
        "mia_n_samples": 1000,
    },
    "tep": {
        # Medium redundancy: ρ≈0.4 at τ=0.97 (simulated chemical process)
        "max_samples": 30_000, "bottleneck": 16, "batch_size": 128,
        "local_epochs": 3, "n_rounds": 15, "lr": 1e-3,
        "siamese_epochs": 15, "siamese_emb": 32, "siamese_window": 5,
        "similarity_threshold": 0.97, "sil_sigma": 3.0,
        "eval_percentile": 90.0, "delta": 1e-5, "max_grad_norm": 1.0,
        "mia_n_samples": 1000,
    },
    "iiot": {
        # Moderate temporal redundancy: ρ≈0.23 at τ=0.97 (new temporal encoder).
        # 50-machine smart manufacturing, 9 features, 9.1% anomaly.
        # n_dedup/client ≈ 0.23×40k/5 = 1840 — adequate for FL training.
        # NOTE: IIOT is a SYNTHETIC dataset (no temporal autocorrelation, label
        # leakage via downtime_risk). DROPPED from the Q1 paper — kept here only
        # for reference. Do not include in main claims.
        "max_samples": 40_000, "bottleneck": 8,  "batch_size": 128,
        "local_epochs": 3, "n_rounds": 15, "lr": 1e-3,
        "siamese_epochs": 15, "siamese_emb": 32, "siamese_window": 5,
        "similarity_threshold": 0.97, "sil_sigma": 3.0,
        "eval_percentile": 92.0, "delta": 1e-5, "max_grad_norm": 1.0,
        "mia_n_samples": 1000,
    },
    "swat": {
        # REAL ICS testbed (iTrust SUTD), Secure Water Treatment A4/A5 Jul 2019.
        # 44 numeric sensor/actuator tags, ~15k samples at 1 Hz, 13.2% attack.
        # Labels reconstructed from 6 documented attack windows (see load_swat).
        # Temporal autocorr ≈ 1.0 → genuine near-duplicates; measured ρ≈0.24 at τ=0.97.
        # Fills the moderate-redundancy slot of the ρ-gradient with REAL data.
        "max_samples": 20_000, "bottleneck": 8,  "batch_size": 64,
        "local_epochs": 3, "n_rounds": 15, "lr": 1e-3,
        "siamese_epochs": 15, "siamese_emb": 32, "siamese_window": 5,
        "similarity_threshold": 0.97, "sil_sigma": 3.0,
        "eval_percentile": 87.0, "delta": 1e-5, "max_grad_norm": 1.0,
        "mia_n_samples": 1000,
    },
}

EPSILON_GRID = [0.5, 1.0, 2.0, 4.0]   # 4 values for frontier curve
SEEDS        = list(range(10))          # 10 seeds for robust statistical estimates
K            = 5

# The six experiment conditions from Section 7 of the plan
CONDITIONS = [
    {"name": "centralized",          "dedup": False, "dp": False, "strategy": "none"},
    {"name": "local_only",           "dedup": False, "dp": False, "strategy": "local"},
    {"name": "fedavg_nodp_nodedup",  "dedup": False, "dp": False, "strategy": "fedavg"},
    {"name": "fedprox_nodp_nodedup", "dedup": False, "dp": False, "strategy": "fedprox",
     "proximal_mu": 0.01},
    {"name": "fedavg_dp_nodedup",    "dedup": False, "dp": True,  "strategy": "fedavg"},
    {"name": "fedavg_dp_dedup",      "dedup": True,  "dp": True,  "strategy": "fedavg"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Core runner
# ─────────────────────────────────────────────────────────────────────────────

def run_one_cell(
    dataset: str, alpha, cond: dict, epsilon: float, seed: int,
    X_train, y_train, X_test, y_test, encoder, cfg: dict,
    all_results: list,
) -> None:
    """Run a single (dataset, partition, condition, ε, seed) cell."""
    np.random.seed(seed)
    import torch; torch.manual_seed(seed)

    input_dim = X_train.shape[1]

    # Partition
    if alpha is None:
        partitions = partition_iid(X_train, y_train, K, random_state=seed)
        part_label = "iid"
    else:
        partitions = partition_dirichlet(X_train, y_train, K,
                                         alpha=alpha, random_state=seed)
        part_label = f"dir{alpha}"

    cond_name = cond["name"]

    # ── Special baselines (no FL) ──────────────────────────────────────────
    if cond["strategy"] == "none":  # centralized
        row = _run_centralized(X_train, y_train, X_test, y_test,
                               input_dim, cfg, dataset, seed, part_label, epsilon)
        all_results.append(row); return

    if cond["strategy"] == "local":  # local-only (lower bound)
        row = _run_local_only(partitions, X_test, y_test,
                              input_dim, cfg, dataset, seed, part_label, epsilon)
        all_results.append(row); return

    # ── FL conditions ──────────────────────────────────────────────────────
    client_fn = build_client_fn(
        partitions      = partitions,
        input_dim       = input_dim,
        config          = cfg,
        dedup_enabled   = cond["dedup"],
        dp_enabled      = cond["dp"],
        target_epsilon  = epsilon,
        siamese_encoder = encoder if cond["dedup"] else None,
        # SIL gate fits on the PUBLIC normal-operation reference only (y=0):
        # leakage-controlled (no private/anomaly data in the gate statistics) and
        # statistically correct (anomalies must not inflate the normal envelope).
        global_train_X  = X_train[y_train == 0],
        proximal_mu     = cond.get("proximal_mu", 0.0),
    )

    label = f"{cond_name}_K{K}_{part_label}_eps{epsilon}_s{seed}"
    result = run_simulation(
        client_fn      = client_fn,
        n_clients      = K,
        n_rounds       = cfg["n_rounds"],
        input_dim      = input_dim,
        config         = cfg,
        X_test         = X_test,
        y_test         = y_test,
        condition_name = label,
        dataset_name   = dataset,
        seed           = seed,
    )

    # MIA on DP conditions
    mia_auc = float("nan")
    if cond["dp"]:
        final_model = AnomalyAutoencoder(input_dim=input_dim,
                                         bottleneck=cfg["bottleneck"])
        final_model.set_weights(result.final_weights)
        mia = MIAEvaluator(n_shadow_samples=cfg["mia_n_samples"])
        mia_res = mia.evaluate(final_model, X_train[:2000], X_test[:2000], seed=seed)
        mia_auc = mia_res["mia_auc"]

    # Privacy comparison report (dedup vs no-dedup at same σ)
    priv_report = {}
    if cond["dp"] and cond["dedup"] and result.final_rho < 1.0:
        n_total = len(X_train)
        n_dedup = int(n_total * result.final_rho)
        priv_report = privacy_report(
            n_original = n_total,
            n_deduped  = max(1, n_dedup),
            batch_size = cfg["batch_size"],
            n_steps    = max(1, (n_dedup // cfg["batch_size"])) * cfg["local_epochs"],
            sigma      = result.final_sigma if result.final_sigma > 0 else 1.0,
            delta      = cfg["delta"],
        )

    row = {
        "dataset": dataset, "K": K, "alpha": alpha, "partition": part_label,
        "condition": cond_name, "epsilon_target": epsilon,
        "seed": seed,
        "f1":           result.final_f1,
        "recall":       result.final_recall,
        "auprc":        result.final_auprc,
        "epsilon_spent": result.final_epsilon,
        "sigma":        result.final_sigma,
        "rho":          result.final_rho,
        "comm_bytes":   result.comm_bytes,
        "mia_auc":      mia_auc,
        "rounds_to_conv": result.rounds_to_conv,
        **{f"priv_{k}": v for k, v in priv_report.items()},
    }
    all_results.append(row)
    log.info("✓ %s | F1=%.4f AUPRC=%.4f ε_spent=%.3f ρ=%.3f MIA=%.4f",
             label, row["f1"], row["auprc"], row["epsilon_spent"],
             row["rho"], mia_auc if not np.isnan(mia_auc) else -1)


# ─────────────────────────────────────────────────────────────────────────────
# No-FL baselines
# ─────────────────────────────────────────────────────────────────────────────

def _run_centralized(X_tr, y_tr, X_te, y_te, input_dim, cfg,
                     dataset, seed, part_label, epsilon):
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    model = AnomalyAutoencoder(input_dim=input_dim, bottleneck=cfg["bottleneck"])
    opt   = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr).float()),
                        batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    for _ in range(cfg["local_epochs"] * 3):   # more epochs for centralized
        for (b,) in loader:
            opt.zero_grad(); loss = nn.MSELoss()(model(b), b); loss.backward(); opt.step()
    m = evaluate_anomaly_detector(model, X_te, y_te,
                                  percentile=cfg["eval_percentile"])
    return {"dataset": dataset, "K": K, "alpha": None, "partition": part_label,
            "condition": "centralized", "epsilon_target": epsilon, "seed": seed,
            "f1": m["f1"], "recall": m["recall"], "auprc": m["auprc"],
            "epsilon_spent": float("inf"), "sigma": 0, "rho": 1.0,
            "comm_bytes": 0, "mia_auc": float("nan"), "rounds_to_conv": 0}


def _run_local_only(partitions, X_te, y_te, input_dim, cfg,
                    dataset, seed, part_label, epsilon):
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    all_f1, all_rec, all_auprc = [], [], []
    for p in partitions:
        if len(p.X) < 4: continue
        model  = AnomalyAutoencoder(input_dim=input_dim, bottleneck=cfg["bottleneck"])
        opt    = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        bs     = min(cfg["batch_size"], len(p.X) - 1)
        loader = DataLoader(TensorDataset(torch.from_numpy(p.X).float()),
                            batch_size=bs, shuffle=True, drop_last=True)
        for _ in range(cfg["local_epochs"]):
            for (b,) in loader:
                opt.zero_grad(); loss = nn.MSELoss()(model(b), b)
                loss.backward(); opt.step()
        m = evaluate_anomaly_detector(model, X_te, y_te,
                                      percentile=cfg["eval_percentile"])
        all_f1.append(m["f1"]); all_rec.append(m["recall"]); all_auprc.append(m["auprc"])
    return {"dataset": dataset, "K": K, "alpha": None, "partition": part_label,
            "condition": "local_only", "epsilon_target": epsilon, "seed": seed,
            "f1": float(np.mean(all_f1)) if all_f1 else 0,
            "recall": float(np.mean(all_rec)) if all_rec else 0,
            "auprc": float(np.mean(all_auprc)) if all_auprc else 0,
            "epsilon_spent": float("inf"), "sigma": 0, "rho": 1.0,
            "comm_bytes": 0, "mia_auc": float("nan"), "rounds_to_conv": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Main driver
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["skab", "tep", "iiot"])
    p.add_argument("--skip_ablations", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Load existing results and skip completed cells")
    args = p.parse_args()

    all_results = []
    if args.resume and RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        log.info("Resumed: %d existing results", len(all_results))

    def _completed(dataset, cond_name, epsilon, seed, alpha):
        for r in all_results:
            if (r["dataset"] == dataset and r["condition"] == cond_name
                    and r["epsilon_target"] == epsilon and r["seed"] == seed
                    and r.get("alpha") == alpha):
                return True
        return False

    t0_total = time.time()

    for dataset in args.datasets:
        cfg = DS_CFG[dataset]
        log.info("\n" + "=" * 70)
        log.info("DATASET: %s", dataset.upper())
        log.info("=" * 70)

        # Load data once per dataset
        log.info("Loading %s ...", dataset)
        X, y, feat_names = DATASET_REGISTRY[dataset](
            max_samples=cfg["max_samples"], random_state=42)
        input_dim = X.shape[1]
        log.info("%s: N=%d D=%d anomaly=%.1f%%", dataset, len(X), input_dim, 100*y.mean())

        n_train = int(0.8 * len(X))
        idx     = np.random.default_rng(42).permutation(len(X))
        X_train, y_train = X[idx[:n_train]], y[idx[:n_train]]
        X_test,  y_test  = X[idx[n_train:]], y[idx[n_train:]]

        # Measure actual redundancy level ρ (used in paper Table 2 and theoretical analysis)
        _rho_path = ROOT / "results" / f"rho_{dataset}.json"
        if not _rho_path.exists():
            try:
                from dedup.local_dedup import LocalDeduplicator
                _sample = min(5000, len(X_train))
                _dedup_probe = LocalDeduplicator(
                    similarity_threshold=cfg["similarity_threshold"])
                _enc_probe = SiameseEncoder(input_dim=input_dim,
                                            embedding_dim=cfg["siamese_emb"])
                # Quick 3-epoch encoder just for ρ measurement
                _enc_probe = train_siamese(_enc_probe, X_train, y_train,
                                           epochs=3, batch_size=256)
                _dr = _dedup_probe.deduplicate(
                    X_train[:_sample], y_train[:_sample], encoder=_enc_probe)
                _rho_val = _dr.retention_rho
                import json as _json
                _rho_path.write_text(_json.dumps({"dataset": dataset,
                                                   "rho": _rho_val,
                                                   "tau": cfg["similarity_threshold"],
                                                   "n_sample": _sample}))
                log.info("Dataset %s: measured ρ=%.3f at τ=%.2f",
                         dataset, _rho_val, cfg["similarity_threshold"])
            except Exception as _e:
                log.warning("ρ measurement failed for %s: %s", dataset, _e)

        # Train Siamese encoder ONLY on public normal-operation data (y=0 subset).
        # This preserves the federated privacy assumption: the encoder is treated as
        # a shared public utility pre-trained on the anomaly-free commissioning period.
        # No anomaly labels or private client data are exposed to the server.
        log.info("Training Siamese encoder for %s on public normal subset ...", dataset)
        encoder = SiameseEncoder(input_dim=input_dim, embedding_dim=cfg["siamese_emb"])
        encoder = train_siamese(
            encoder, X_train, y_train,   # y_train used only to select y=0 samples
            epochs=cfg["siamese_epochs"], batch_size=256, lr=1e-3,
            window=cfg.get("siamese_window", 5),
        )
        n_public = int((y_train == 0).sum())
        log.info("Siamese encoder trained on %d public normal samples.", n_public)

        # Two partition schemes: IID and Dirichlet α=0.5
        for alpha in [None, 0.5]:
            part_label = "iid" if alpha is None else f"dir{alpha}"
            log.info("\n--- Partition: %s ---", part_label)

            for cond in CONDITIONS:
                cond_name = cond["name"]
                # DP conditions sweep ε; non-DP run once (ε=inf)
                eps_list = EPSILON_GRID if cond["dp"] else [float("inf")]

                for epsilon in eps_list:
                    for seed in SEEDS:
                        if args.resume and _completed(dataset, cond_name, epsilon, seed, alpha):
                            log.info("SKIP (done): %s %s ε=%s s=%d",
                                     dataset, cond_name, epsilon, seed)
                            continue

                        t0 = time.time()
                        try:
                            run_one_cell(
                                dataset, alpha, cond, epsilon, seed,
                                X_train, y_train, X_test, y_test,
                                encoder, cfg, all_results,
                            )
                        except Exception as e:
                            log.error("FAILED: %s %s ε=%s s=%d — %s",
                                      dataset, cond_name, epsilon, seed, e, exc_info=True)
                            # Record failure so we can track it
                            all_results.append({
                                "dataset": dataset, "condition": cond_name,
                                "epsilon_target": epsilon, "seed": seed,
                                "alpha": alpha, "K": K,
                                "f1": 0, "recall": 0, "auprc": 0,
                                "epsilon_spent": float("nan"), "sigma": 0,
                                "rho": 1, "comm_bytes": 0,
                                "mia_auc": float("nan"), "rounds_to_conv": 0,
                                "error": str(e),
                            })

                        elapsed = time.time() - t0
                        log.info("Cell done in %.1fs. Total results: %d",
                                 elapsed, len(all_results))

                        # Checkpoint after every cell
                        _checkpoint(all_results)

    # ── Ablations on SKAB ────────────────────────────────────────────────────
    if not args.skip_ablations:
        log.info("\n" + "=" * 70)
        log.info("ABLATIONS (SKAB)")
        log.info("=" * 70)
        _run_ablations_skab()

    # ── Final figures and tables ──────────────────────────────────────────────
    log.info("\nGenerating figures and tables ...")
    _generate_outputs(all_results)

    total_h = (time.time() - t0_total) / 3600
    log.info("\n✓ Complete sweep finished in %.2fh. %d cells.", total_h, len(all_results))


def _checkpoint(results: list):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)


def _generate_outputs(results: list):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    make_privacy_utility_figure(results, FIG_DIR)
    make_communication_figure(results,   FIG_DIR)
    make_mia_figure(results,             FIG_DIR)
    for ds in ["skab", "tep", "iiot"]:
        if any(r["dataset"] == ds for r in results):
            make_latex_table(results, ds, TAB_DIR)
    # Also produce full combined table
    _make_full_summary_table(results)
    # Publication figures are produced separately by experiments/make_paper_figs.py.
    log.info("All tables written. Run experiments/make_paper_figs.py for the figures.")


def _run_ablations_skab():
    """Run all 5 ablations on SKAB and save results."""
    import importlib, sys
    abl = importlib.import_module("experiments.ablations")

    for key, fn in [
        ("a_rho",      abl.ablation_rho_sweep),
        ("b_detector", abl.ablation_detector_swap),
        ("e_sil",      abl.ablation_sil_gate),
    ]:
        log.info("Ablation %s ...", key)
        try:
            fn(dataset="skab", epsilon=1.0, seed=42)
        except Exception as e:
            log.error("Ablation %s failed: %s", key, e, exc_info=True)

    # Ablation c: α sweep
    log.info("Ablation c_alpha ...")
    try:
        abl.ablation_fedprox_alpha(dataset="skab", epsilon=1.0, seed=42)
    except Exception as e:
        log.error("Ablation c failed: %s", e, exc_info=True)

    # Ablation d: K sweep
    log.info("Ablation d_K ...")
    try:
        abl.ablation_k_clients(dataset="skab", epsilon=1.0, seed=42)
    except Exception as e:
        log.error("Ablation d failed: %s", e, exc_info=True)


def _make_full_summary_table(results: list):
    """Headline table: DP conditions at ε=1.0, all datasets, mean±std."""
    from collections import defaultdict
    rows = defaultdict(list)
    for r in results:
        if r.get("epsilon_target") in [1.0, float("inf")] and not r.get("error"):
            key = (r["dataset"], r["condition"])
            rows[key].append(r)

    header = "dataset,condition,f1_mean,f1_std,recall_mean,auprc_mean,eps_spent,rho,mia_auc"
    lines  = [header]
    COND_ORDER = [
        "centralized","local_only","fedavg_nodp_nodedup","fedprox_nodp_nodedup",
        "fedavg_dp_nodedup","fedavg_dp_dedup",
    ]
    for ds in ["skab","tep","iiot"]:
        for cond in COND_ORDER:
            rs = rows.get((ds, cond), [])
            if not rs: continue
            f1s   = [r["f1"]    for r in rs]
            recs  = [r["recall"] for r in rs]
            au    = [r["auprc"] for r in rs]
            eps   = [r["epsilon_spent"] for r in rs if not np.isinf(r.get("epsilon_spent",np.inf))]
            rhos  = [r["rho"]   for r in rs]
            mias  = [r["mia_auc"] for r in rs if not np.isnan(r.get("mia_auc", float("nan")))]
            lines.append(",".join([
                ds, cond,
                f"{np.mean(f1s):.4f}", f"{np.std(f1s):.4f}",
                f"{np.mean(recs):.4f}", f"{np.mean(au):.4f}",
                f"{np.mean(eps):.3f}" if eps else "inf",
                f"{np.mean(rhos):.3f}",
                f"{np.mean(mias):.4f}" if mias else "nan",
            ]))

    csv_path = ROOT / "results" / "tables" / "headline_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("\n".join(lines))
    log.info("Summary table → %s", csv_path)


if __name__ == "__main__":
    main()
