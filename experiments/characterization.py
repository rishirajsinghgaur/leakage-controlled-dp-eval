r"""
DEFINITIVE characterization experiment (core evidence for the Q1 paper).
Centralized DP reconstruction anomaly detector trained on NORMAL-ONLY data; compare
selection methods at MATCHED budget to show selection-INVARIANCE (it's the subset-size
effect, not the selection cleverness):
  full   : all normal data
  random : uniform random subsample (strong baseline; arXiv 2302.06960)
  tdedup : adaptive sequential temporal dedup
  fps    : farthest-point (k-center) diversity coreset
SKAB+SWaT+TEP, eps {0.5,1,2,4}, 5 seeds. Real outputs -> results/characterization.json.
"""
import sys, json, logging
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.ERROR)
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
_OUTP = ROOT / "results" / "characterization.json"
_done = json.load(open(_OUTP)) if _OUTP.exists() else []   # resume: keep prior rows
_seen = {(r["dataset"], r["mode"], r["epsilon"], r["seed"]) for r in _done}


def fps_coreset(E, budget):
    n = len(E); budget = min(budget, n); sel = [0]; d = 1.0 - E @ E[0]
    for _ in range(1, budget):
        i = int(np.argmax(d)); sel.append(i); d = np.minimum(d, 1.0 - E @ E[i])
    return np.array(sorted(sel))


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
    # Load from npz cache when present (avoids pyarrow-after-torch segfault on parquet ds).
    cpath = ROOT / "results" / f"cache_{ds}.npz"
    if cpath.exists():
        z = np.load(cpath); return z["X"].astype("float32"), z["y"].astype("int64")
    X, y, _ = DATASET_REGISTRY[ds](max_samples=cfg["max_samples"], random_state=42)
    return X, y


def main():
    out = list(_done)   # resume from prior rows
    for ds in _DATASETS:
        cfg = dict(DS_CFG[ds]); cfg["siamese_epochs"] = min(cfg["siamese_epochs"], 6)  # encoder only selects tdedup/fps subsets; cap for tractability
        X, y = _load(ds, cfg)
        idx = np.arange(len(X)); tr, te = train_test_split(idx, test_size=0.2, random_state=0, stratify=y); tr = np.sort(tr)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        enc = SiameseEncoder(Xtr.shape[1], cfg["siamese_emb"]); enc = train_siamese(enc, Xtr, ytr, epochs=cfg["siamese_epochs"], window=cfg["siamese_window"])
        Xn = Xtr[ytr == 0]
        if len(Xn) > 6000: Xn = Xn[:6000]
        E = enc.encode(Xn)
        kept_t, _ = seq_temporal_dedup(Xn, enc, keep_quantile=0.5); Xd = Xn[kept_t]; budget = len(Xd)
        Xf = Xn[fps_coreset(E, budget)]
        print(f"{ds}: normal={len(Xn)} budget={budget} (rho={budget/len(Xn):.3f})", flush=True)
        for seed in SEEDS:
            rng = np.random.default_rng(1000 + seed)
            Xr = Xn[np.sort(rng.choice(len(Xn), size=budget, replace=False))]   # random, matched budget
            for eps in EPS:
                for mode, Xt in [("full", Xn), ("random", Xr), ("tdedup", Xd), ("fps", Xf)]:
                    if (ds, mode, eps, seed) in _seen:   # resume: skip completed cells
                        continue
                    f1, ap = dp_eval(Xt, Xte, yte, cfg, eps, seed)
                    out.append({"dataset": ds, "mode": mode, "epsilon": eps, "seed": seed, "f1": round(f1, 4), "auprc": round(ap, 4), "n": len(Xt)})
                    print(f"  {ds} {mode} eps={eps} s={seed} F1={f1:.4f} AUPRC={ap:.4f}", flush=True)
                    _OUTP.write_text(json.dumps(out, indent=2))
    print("\n=== CHARACTERIZATION SUMMARY (F1 mean +/- std) ===")
    for ds in sorted(set(r["dataset"] for r in out)):
        for eps in EPS:
            s = []
            for mode in ["full", "random", "tdedup", "fps"]:
                v = [x["f1"] for x in out if x["dataset"] == ds and x["mode"] == mode and x["epsilon"] == eps]
                s.append(f"{mode}={np.mean(v):.3f}+/-{np.std(v):.3f}" if v else f"{mode}=NA")
            print(f"{ds} eps={eps}: " + "  ".join(s))


if __name__ == "__main__":
    main()
