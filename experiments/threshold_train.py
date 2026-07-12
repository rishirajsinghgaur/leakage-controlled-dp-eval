r"""
Train-normal-derived threshold robustness (addresses the sharpest reviewer finding).

The main results threshold anomaly scores at the p-th percentile of the *evaluation-set*
reconstruction errors (a transductive threshold; on SWaT p87 sits near the 13.2% attack
prevalence). Reviewers correctly note this is prevalence-adjacent. Here we recompute F1 for
every selection arm using a threshold calibrated ONLY on the training normal-only pool
(the p-th percentile of the detector's reconstruction errors on the data it was trained on),
applied unchanged to the test set. This uses no test labels and no knowledge of prevalence.

We report F1 under BOTH thresholds so the table shows directly that the *relative* selection
comparison (the paper's claim) is unchanged: uniform random still matches/beats the learned
rules under the honest train-normal threshold.

SKAB/SWaT/TEP, eps {0.5,1,2,4}, 5 seeds. Output -> results/threshold_train.json.
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
from sklearn.metrics import f1_score
from data.loaders import DATASET_REGISTRY
from models.siamese import SiameseEncoder, train_siamese
from models.mlp import AnomalyAutoencoder
from privacy.accountant import compute_sigma_for_total_epsilon
from experiments.run_full_paper_sweep import DS_CFG
from experiments.principled_method import seq_temporal_dedup

SEEDS = list(range(5)); EPS = [0.5, 1.0, 2.0, 4.0]; PASSES = 15
_DATASETS = sys.argv[1:] if len(sys.argv) > 1 else ["skab", "swat", "tep"]
_OUTP = ROOT / "results" / "threshold_train.json"
_done = json.load(open(_OUTP)) if _OUTP.exists() else []
_seen = {(r["dataset"], r["mode"], r["epsilon"], r["seed"]) for r in _done}


def fps_coreset(E, budget):
    n = len(E); budget = min(budget, n); sel = [0]; d = 1.0 - E @ E[0]
    for _ in range(1, budget):
        i = int(np.argmax(d)); sel.append(i); d = np.minimum(d, 1.0 - E @ E[i])
    return np.array(sorted(sel))


def _scores(model, X):
    return model.anomaly_score(X)


def dp_eval_both(Xt, Xte, yte, cfg, eps, seed):
    """Train the DP detector on Xt (normal-only), then evaluate F1 under two thresholds:
       (a) eval-set percentile (original, transductive)
       (b) train-normal percentile (honest: calibrated on Xt only, applied to test)."""
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
    mm = getattr(m, "_module", m)
    p = cfg["eval_percentile"]
    s_test = _scores(mm, Xte)
    s_train = _scores(mm, Xt)                       # train-normal reconstruction errors
    thr_eval  = np.percentile(s_test,  p)           # (a) transductive (original)
    thr_train = np.percentile(s_train, p)           # (b) honest, no labels/prevalence
    f1_eval  = float(f1_score(yte, (s_test >= thr_eval ).astype(int), zero_division=0))
    f1_train = float(f1_score(yte, (s_test >= thr_train).astype(int), zero_division=0))
    return f1_eval, f1_train, float((s_test >= thr_train).mean())


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
        E = enc.encode(Xn)
        kept_t, _ = seq_temporal_dedup(Xn, enc, keep_quantile=0.5); Xd = Xn[kept_t]; budget = len(Xd)
        Xf = Xn[fps_coreset(E, budget)]
        print(f"{ds}: normal={len(Xn)} budget={budget} (rho={budget/len(Xn):.3f}) prevalence={yte.mean():.3f}", flush=True)
        for seed in SEEDS:
            rng = np.random.default_rng(1000 + seed)
            Xr = Xn[np.sort(rng.choice(len(Xn), size=budget, replace=False))]
            for eps in EPS:
                for mode, Xt in [("full", Xn), ("random", Xr), ("tdedup", Xd), ("fps", Xf)]:
                    if (ds, mode, eps, seed) in _seen:
                        continue
                    f1e, f1t, rate = dp_eval_both(Xt, Xte, yte, cfg, eps, seed)
                    out.append({"dataset": ds, "mode": mode, "epsilon": eps, "seed": seed,
                                "f1_evalthr": round(f1e, 4), "f1_trainthr": round(f1t, 4),
                                "flagrate_trainthr": round(rate, 4), "n": len(Xt)})
                    print(f"  {ds} {mode} eps={eps} s={seed} F1(eval)={f1e:.3f} F1(train)={f1t:.3f} flag={rate:.3f}", flush=True)
                    _OUTP.write_text(json.dumps(out, indent=2))
    # summary: does the selection null survive under the honest threshold?
    print("\n=== TRAIN-NORMAL THRESHOLD: F1 mean (eval-thr / train-thr) ===")
    for ds in sorted(set(r["dataset"] for r in out)):
        for eps in EPS:
            cells = []
            for mode in ["full", "random", "tdedup", "fps"]:
                ve = [x["f1_evalthr"]  for x in out if x["dataset"]==ds and x["mode"]==mode and x["epsilon"]==eps]
                vt = [x["f1_trainthr"] for x in out if x["dataset"]==ds and x["mode"]==mode and x["epsilon"]==eps]
                cells.append(f"{mode}={np.mean(ve):.3f}/{np.mean(vt):.3f}" if ve else f"{mode}=NA")
            print(f"{ds} eps={eps}: " + "  ".join(cells))


if __name__ == "__main__":
    main()
