r"""
DP-aware selection baseline (addresses Review #1's "does a selector that BUDGETS its own
privacy cost beat random?"). This is the constructive counterpart to the null: instead of
spending no privacy on selection (the public-data protocol, eps_sel=0), here the selector
spends eps_sel on a genuinely differentially private selection and trains on the remaining
eps_train = eps_total - eps_sel. We compare it to uniform random (eps_sel=0, eps_train=eps_total)
at MATCHED total budget. The suppression theorem predicts it cannot win.

DP selection mechanism (eps_sel-DP, composed then with eps_train training):
  1. Clip each normal record to L2 norm <= C (C = 90th percentile of row norms).
  2. Private mean mu_hat via the Gaussian mechanism at (eps_sel/2, delta); L2 sensitivity 2C/n.
  3. Score s_i = ||x_i - mu_hat|| (clipped to [0, 2C]); release the noisy score vector
     s_i + Laplace(2C / (eps_sel/2)). Since neighbouring datasets differ in ONE record, the
     released score vector has L1 sensitivity 2C, so this is (eps_sel/2)-DP; selecting the
     top-`budget` records by noisy score is post-processing.
  4. Total selection is (eps_sel)-DP; training at eps_train composes to eps_total (basic
     sequential composition -- a conservative bound that charges the selection its full cost,
     matching the suppression theorem).

We select the FARTHEST-from-mean records (a privacy-spending diversity/hard-example coreset,
the "smart" choice). Datasets: SKAB, SWaT. eps_total {1,2,4}, eps_sel fraction {0.25,0.5},
5 seeds. Output -> results/dp_aware_selection.json.
"""
import sys, json, math
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.stdout.reconfigure(encoding="utf-8")
import logging; logging.basicConfig(level=logging.ERROR)
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from data.loaders import DATASET_REGISTRY
from models.siamese import SiameseEncoder, train_siamese
from models.mlp import AnomalyAutoencoder, evaluate_anomaly_detector
from privacy.accountant import compute_sigma_for_total_epsilon
from experiments.run_full_paper_sweep import DS_CFG
from experiments.principled_method import seq_temporal_dedup

SEEDS = list(range(5)); EPS_TOTAL = [1.0, 2.0, 4.0]; FRACS = [0.25, 0.5]; PASSES = 15
DELTA_SEL = 1e-5
_DATASETS = sys.argv[1:] if len(sys.argv) > 1 else ["skab", "swat"]
_OUTP = ROOT / "results" / "dp_aware_selection.json"
_done = json.load(open(_OUTP)) if _OUTP.exists() else []
_seen = {(r["dataset"], r["mode"], r["eps_total"], r["seed"]) for r in _done}


def dp_select_far(Xn, budget, eps_sel, seed):
    """(eps_sel)-DP selection of `budget` records farthest from the private mean."""
    rng = np.random.default_rng(10_000 + seed)
    norms = np.linalg.norm(Xn, axis=1)
    C = float(np.percentile(norms, 90))                       # clip bound
    Xc = Xn * np.minimum(1.0, C / np.maximum(norms, 1e-9))[:, None]
    n = len(Xn)
    # (eps_sel/2)-DP Gaussian-mechanism mean; L2 sensitivity of the mean = 2C/n
    e_mean = eps_sel / 2.0
    sigma_mean = (2 * C / n) * math.sqrt(2 * math.log(1.25 / DELTA_SEL)) / e_mean
    mu = Xc.mean(0) + rng.normal(0, sigma_mean, size=Xc.shape[1])
    # scores = distance to private mean, clipped; release noisy scores (eps_sel/2)-DP
    s = np.clip(np.linalg.norm(Xc - mu, axis=1), 0, 2 * C)
    e_score = eps_sel / 2.0
    lap_scale = (2 * C) / e_score                             # L1 sensitivity 2C (one record -> one score)
    s_noisy = s + rng.laplace(0, lap_scale, size=n)
    return np.sort(np.argsort(s_noisy)[-budget:])             # top-budget farthest (post-processing)


def dp_train_eval(Xt, Xte, yte, cfg, eps_train, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(Xt); B = min(cfg["batch_size"], max(2, n - 1)); steps = max(1, n // B) * PASSES
    sig = compute_sigma_for_total_epsilon(eps_train, n, B, steps, 1, cfg["delta"])
    from opacus import PrivacyEngine
    m = AnomalyAutoencoder(Xt.shape[1], cfg["bottleneck"]); opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
    ld = DataLoader(TensorDataset(torch.from_numpy(Xt).float()), batch_size=B, shuffle=True, drop_last=True)
    m, opt, ld = PrivacyEngine().make_private(module=m, optimizer=opt, data_loader=ld, noise_multiplier=sig, max_grad_norm=cfg["max_grad_norm"])
    lf = nn.MSELoss(); m.train()
    for _ in range(PASSES):
        for (b,) in ld:
            opt.zero_grad(); lf(m(b), b).backward(); opt.step()
    mm = getattr(m, "_module", m); met = evaluate_anomaly_detector(mm, Xte, yte, percentile=cfg["eval_percentile"])
    return met["f1"]


def _load(ds, cfg):
    cpath = ROOT / "results" / f"cache_{ds}.npz"
    if cpath.exists():
        z = np.load(cpath); return z["X"].astype("float32"), z["y"].astype("int64")
    X, y, _ = DATASET_REGISTRY[ds](max_samples=cfg["max_samples"], random_state=42)
    return X, y


def main():
    out = list(_done)
    for ds in _DATASETS:
        cfg = dict(DS_CFG[ds]); cfg["siamese_epochs"] = min(cfg["siamese_epochs"], 6)
        X, y = _load(ds, cfg)
        idx = np.arange(len(X)); tr, te = train_test_split(idx, test_size=0.2, random_state=0, stratify=y); tr = np.sort(tr)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        enc = SiameseEncoder(Xtr.shape[1], cfg["siamese_emb"]); enc = train_siamese(enc, Xtr, ytr, epochs=cfg["siamese_epochs"], window=cfg["siamese_window"])
        Xn = Xtr[ytr == 0]
        if len(Xn) > 6000: Xn = Xn[:6000]
        kept_t, _ = seq_temporal_dedup(Xn, enc, keep_quantile=0.5); budget = len(kept_t)
        print(f"{ds}: normal={len(Xn)} budget={budget} (rho={budget/len(Xn):.3f})", flush=True)
        for seed in SEEDS:
            rng = np.random.default_rng(1000 + seed)
            for eps_total in EPS_TOTAL:
                arms = [("random", None, Xn[np.sort(rng.choice(len(Xn), size=budget, replace=False))], eps_total)]
                for f in FRACS:
                    e_sel = round(f * eps_total, 4); e_tr = round(eps_total - e_sel, 4)
                    sub = Xn[dp_select_far(Xn, budget, e_sel, seed)]
                    arms.append((f"dpfar{int(f*100)}", e_sel, sub, e_tr))
                for mode, e_sel, Xt, e_tr in arms:
                    if (ds, mode, eps_total, seed) in _seen:
                        continue
                    f1 = dp_train_eval(Xt, Xte, yte, cfg, e_tr, seed)
                    out.append({"dataset": ds, "mode": mode, "eps_total": eps_total,
                                "eps_sel": e_sel, "eps_train": e_tr, "seed": seed,
                                "f1": round(f1, 4), "n": len(Xt)})
                    print(f"  {ds} {mode} eps_tot={eps_total} (sel={e_sel},tr={e_tr}) s={seed} F1={f1:.4f}", flush=True)
                    _OUTP.write_text(json.dumps(out, indent=2))
    print("\n=== DP-AWARE SELECTION (F1 mean +/- std) vs random at matched total eps ===")
    for ds in sorted(set(r["dataset"] for r in out)):
        for et in EPS_TOTAL:
            cells = []
            for mode in ["random", "dpfar25", "dpfar50"]:
                v = [x["f1"] for x in out if x["dataset"]==ds and x["mode"]==mode and x["eps_total"]==et]
                cells.append(f"{mode}={np.mean(v):.3f}±{np.std(v):.3f}" if v else f"{mode}=NA")
            print(f"{ds} eps_total={et}: " + "  ".join(cells))


if __name__ == "__main__":
    main()
