r"""
Public-reference-size ablation (addresses "the 6,000 cap is unjustified / not ablated").

Selection statistics (the similarity encoder and the temporal/diversity subsets) are fit on
a public normal pool that the main experiments cap at 6,000 records. Here we vary that cap
C in {2000, 6000, full} and repeat the selection comparison (full/random/tdedup/fps) at each,
to show the selection null does not depend on the reference size. Everything else is
identical to characterization.py.

SKAB/SWaT/TEP, eps {1,2} (two budgets suffice to show invariance), 5 seeds.
Output -> results/reference_size.json.
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

SEEDS = list(range(5)); EPS = [1.0, 2.0]; PASSES = 15; CAPS = [2000, 6000, 999999]
_DATASETS = sys.argv[1:] if len(sys.argv) > 1 else ["skab", "swat", "tep"]
_OUTP = ROOT / "results" / "reference_size.json"
_done = json.load(open(_OUTP)) if _OUTP.exists() else []
_seen = {(r["dataset"], r["cap"], r["mode"], r["epsilon"], r["seed"]) for r in _done}


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
        Xn_full = Xtr[ytr == 0]
        for cap in CAPS:
            Xn = Xn_full[:cap] if len(Xn_full) > cap else Xn_full
            realcap = len(Xn)
            enc = SiameseEncoder(Xtr.shape[1], cfg["siamese_emb"]); enc = train_siamese(enc, Xtr, ytr, epochs=cfg["siamese_epochs"], window=cfg["siamese_window"])
            E = enc.encode(Xn)
            kept_t, _ = seq_temporal_dedup(Xn, enc, keep_quantile=0.5); Xd = Xn[kept_t]; budget = len(Xd)
            Xf = Xn[fps_coreset(E, budget)]
            print(f"{ds} cap={cap} (real={realcap}): budget={budget} rho={budget/realcap:.3f}", flush=True)
            for seed in SEEDS:
                rng = np.random.default_rng(1000 + seed)
                Xr = Xn[np.sort(rng.choice(realcap, size=budget, replace=False))]
                for eps in EPS:
                    for mode, Xt in [("full", Xn), ("random", Xr), ("tdedup", Xd), ("fps", Xf)]:
                        if (ds, cap, mode, eps, seed) in _seen:
                            continue
                        f1 = dp_eval(Xt, Xte, yte, cfg, eps, seed)
                        out.append({"dataset": ds, "cap": cap, "realcap": realcap, "mode": mode,
                                    "epsilon": eps, "seed": seed, "f1": round(f1, 4), "n": len(Xt)})
                        print(f"  {ds} cap={cap} {mode} eps={eps} s={seed} F1={f1:.4f}", flush=True)
                        _OUTP.write_text(json.dumps(out, indent=2))
    print("\n=== REFERENCE-SIZE SWEEP (F1 mean) ===")
    for ds in sorted(set(r["dataset"] for r in out)):
        for cap in sorted(set(r["cap"] for r in out if r["dataset"]==ds)):
            for eps in EPS:
                cells = []
                for mode in ["full", "random", "tdedup", "fps"]:
                    v = [x["f1"] for x in out if x["dataset"]==ds and x["cap"]==cap and x["mode"]==mode and x["epsilon"]==eps]
                    cells.append(f"{mode}={np.mean(v):.3f}" if v else f"{mode}=NA")
                print(f"{ds} cap={cap} eps={eps}: " + "  ".join(cells))


if __name__ == "__main__":
    main()
