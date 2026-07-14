r"""Independent-re-split robustness check (reviewer Issue #1: pseudoreplication).

The main selection comparison fixes one 80/20 split (random_state=0) and varies only
training seeds, so its significance claims are, strictly, conditional on that split. Here we
re-draw the label-stratified 80/20 split itself K times (independent split seeds) and re-run
the full/random/tdedup/fps comparison on each, to test whether the null ordering
(no learned selector beats equal-budget random) is split-invariant. Identical code path as
experiments/characterization.py. Resume-safe. Writes results/resplit_robustness.json.
"""
import sys, json, logging
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.ERROR)
from sklearn.model_selection import train_test_split
from models.siamese import SiameseEncoder, train_siamese
from experiments.characterization import dp_eval, fps_coreset, _load
from experiments.run_full_paper_sweep import DS_CFG
from experiments.principled_method import seq_temporal_dedup

SPLIT_SEEDS = [0, 1, 2, 3, 4]      # 5 independent train/test partitions
TRAIN_SEEDS = [0, 1, 2]            # training stochasticity within each split
EPS = 2.0                          # representative budget
DATASETS = ["swat", "skab"]
OUT = ROOT / "results" / "resplit_robustness.json"


def main():
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {(r["dataset"], r["split_seed"], r["mode"], r["seed"]) for r in out}
    for ds in DATASETS:
        cfg = dict(DS_CFG[ds]); cfg["siamese_epochs"] = min(cfg["siamese_epochs"], 6)
        X, y = _load(ds, cfg)
        for ss in SPLIT_SEEDS:
            idx = np.arange(len(X))
            tr, te = train_test_split(idx, test_size=0.2, random_state=ss, stratify=y); tr = np.sort(tr)
            Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
            enc = SiameseEncoder(Xtr.shape[1], cfg["siamese_emb"])
            enc = train_siamese(enc, Xtr, ytr, epochs=cfg["siamese_epochs"], window=cfg["siamese_window"])
            Xn = Xtr[ytr == 0]
            if len(Xn) > 6000: Xn = Xn[:6000]
            E = enc.encode(Xn)
            kept_t, _ = seq_temporal_dedup(Xn, enc, keep_quantile=0.5); Xd = Xn[kept_t]; budget = len(Xd)
            Xf = Xn[fps_coreset(E, budget)]
            print(f"[{ds}] split={ss}: normal={len(Xn)} budget={budget} (rho={budget/len(Xn):.3f})", flush=True)
            for seed in TRAIN_SEEDS:
                rng = np.random.default_rng(1000 + seed)
                Xr = Xn[np.sort(rng.choice(len(Xn), size=budget, replace=False))]
                for mode, Xt in [("full", Xn), ("random", Xr), ("tdedup", Xd), ("fps", Xf)]:
                    if (ds, ss, mode, seed) in done:
                        continue
                    f1, ap = dp_eval(Xt, Xte, yte, cfg, EPS, seed)
                    out.append({"dataset": ds, "split_seed": ss, "mode": mode, "seed": seed,
                                "epsilon": EPS, "f1": round(float(f1), 4)})
                    OUT.write_text(json.dumps(out, indent=2))
                    print(f"  {ds} split={ss} {mode} s={seed}: f1={f1:.4f}", flush=True)
    # summary: per split, does any learned selector beat random (mean over train seeds)?
    print("\n=== SPLIT-INVARIANCE SUMMARY (eps=2) ===", flush=True)
    for ds in DATASETS:
        print(f"[{ds}]", flush=True); beats = 0
        for ss in SPLIT_SEEDS:
            cell = {m: np.mean([r["f1"] for r in out if r["dataset"] == ds and r["split_seed"] == ss and r["mode"] == m])
                    for m in ["full", "random", "tdedup", "fps"]}
            win = max(cell["tdedup"], cell["fps"]) > cell["random"] + 1e-9
            beats += win
            print(f"  split {ss}: full={cell['full']:.3f} random={cell['random']:.3f} "
                  f"tdedup={cell['tdedup']:.3f} fps={cell['fps']:.3f}  learned>random={win}", flush=True)
        print(f"  -> learned beats random in {beats}/{len(SPLIT_SEEDS)} splits", flush=True)
    print("\nDONE resplit_robustness.", flush=True)


if __name__ == "__main__":
    main()
