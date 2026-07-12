r"""
Third selection family: loss/score-based data pruning (EL2N / GraNd analogue).

Reviewers note the selection null is established over only two learned rules (temporal
deduplication, diversity coreset). This adds a third, methodologically distinct family:
loss-based pruning. A small non-DP probe autoencoder is trained on the PUBLIC normal pool;
each normal record is scored by its reconstruction error, and the subset is chosen by that
score. This is the reconstruction analogue of EL2N/GraNd data pruning (Paul et al. 2021),
which selects examples by their loss/gradient magnitude. We test both directions:
  lossprune_hard : keep the highest-error (hardest / most informative) records  [EL2N-style]
  lossprune_easy : keep the lowest-error (most prototypical) records
Both are compared to full and matched-size uniform random at equal budget, so any advantage
would show up as beating random. Selection statistics use only public normal data (eps_sel=0).

SKAB/SWaT/TEP, eps {0.5,1,2,4}, 5 seeds. Output -> results/loss_selection.json.
"""
import sys, json
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

SEEDS = list(range(5)); EPS = [0.5, 1.0, 2.0, 4.0]; PASSES = 15
_DATASETS = sys.argv[1:] if len(sys.argv) > 1 else ["skab", "swat", "tep"]
_OUTP = ROOT / "results" / "loss_selection.json"
_done = json.load(open(_OUTP)) if _OUTP.exists() else []
_seen = {(r["dataset"], r["mode"], r["epsilon"], r["seed"]) for r in _done}


def fps_coreset(E, budget):
    n = len(E); budget = min(budget, n); sel = [0]; d = 1.0 - E @ E[0]
    for _ in range(1, budget):
        i = int(np.argmax(d)); sel.append(i); d = np.minimum(d, 1.0 - E @ E[i])
    return np.array(sorted(sel))


def probe_losses(Xn, bottleneck, epochs=8, seed=0):
    """Non-DP probe autoencoder on public normal pool -> per-record reconstruction error."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = AnomalyAutoencoder(Xn.shape[1], bottleneck); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    ld = DataLoader(TensorDataset(torch.from_numpy(Xn).float()), batch_size=128, shuffle=True, drop_last=True)
    lf = nn.MSELoss(); m.train()
    for _ in range(epochs):
        for (b,) in ld:
            opt.zero_grad(); lf(m(b), b).backward(); opt.step()
    m.eval()
    return m.anomaly_score(Xn)


def dp_eval(Xt, Xte, yte, cfg, eps, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(Xt); B = min(cfg["batch_size"], max(2, n - 1)); steps = max(1, n // B) * PASSES
    sig = compute_sigma_for_total_epsilon(eps, n, B, steps, 1, cfg["delta"])
    from opacus import PrivacyEngine
    m = AnomalyAutoencoder(Xt.shape[1], cfg["bottleneck"]); opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
    ld = DataLoader(TensorDataset(torch.from_numpy(Xt).float()), batch_size=B, shuffle=True, drop_last=True)
    m, opt, ld = PrivacyEngine().make_private(module=m, optimizer=opt, data_loader=ld, noise_multiplier=sig, max_grad_norm=cfg["max_grad_norm"])
    lf = nn.MSELoss(); m.train()
    for _ in range(PASSES):
        for (b,) in ld:
            opt.zero_grad(); lf(m(b), b).backward(); opt.step()
    mm = getattr(m, "_module", m); met = evaluate_anomaly_detector(mm, Xte, yte, percentile=cfg["eval_percentile"])
    return met["f1"], met["auprc"]


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
        Xn = Xtr[ytr == 0]
        if len(Xn) > 6000: Xn = Xn[:6000]
        # budget matches the temporal-dedup keep count, exactly as in characterization.py
        enc = SiameseEncoder(Xtr.shape[1], cfg["siamese_emb"]); enc = train_siamese(enc, Xtr, ytr, epochs=cfg["siamese_epochs"], window=cfg["siamese_window"])
        kept_t, _ = seq_temporal_dedup(Xn, enc, keep_quantile=0.5); budget = len(kept_t)
        # loss-based subsets from a public-data probe (eps_sel = 0)
        L = probe_losses(Xn, cfg["bottleneck"], epochs=8, seed=0)
        order = np.argsort(L)                       # ascending error
        idx_easy = np.sort(order[:budget])          # lowest error
        idx_hard = np.sort(order[-budget:])         # highest error
        Xeasy, Xhard = Xn[idx_easy], Xn[idx_hard]
        print(f"{ds}: normal={len(Xn)} budget={budget} (rho={budget/len(Xn):.3f})", flush=True)
        for seed in SEEDS:
            rng = np.random.default_rng(1000 + seed)
            Xr = Xn[np.sort(rng.choice(len(Xn), size=budget, replace=False))]
            for eps in EPS:
                for mode, Xt in [("full", Xn), ("random", Xr),
                                 ("lossprune_hard", Xhard), ("lossprune_easy", Xeasy)]:
                    if (ds, mode, eps, seed) in _seen:
                        continue
                    f1, ap = dp_eval(Xt, Xte, yte, cfg, eps, seed)
                    out.append({"dataset": ds, "mode": mode, "epsilon": eps, "seed": seed,
                                "f1": round(f1, 4), "auprc": round(ap, 4), "n": len(Xt)})
                    print(f"  {ds} {mode} eps={eps} s={seed} F1={f1:.4f}", flush=True)
                    _OUTP.write_text(json.dumps(out, indent=2))
    print("\n=== LOSS-BASED PRUNING (F1 mean +/- std) ===")
    for ds in sorted(set(r["dataset"] for r in out)):
        for eps in EPS:
            cells = []
            for mode in ["full", "random", "lossprune_hard", "lossprune_easy"]:
                v = [x["f1"] for x in out if x["dataset"]==ds and x["mode"]==mode and x["epsilon"]==eps]
                cells.append(f"{mode}={np.mean(v):.3f}+/-{np.std(v):.3f}" if v else f"{mode}=NA")
            print(f"{ds} eps={eps}: " + "  ".join(cells))


if __name__ == "__main__":
    main()
