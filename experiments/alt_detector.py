r"""
Reviewer M1: the null should not be specific to the reconstruction autoencoder.
We repeat the selection comparison (full/random/tdedup/fps, matched per-client budget)
with TWO other detector families — Isolation Forest and One-Class SVM — trained on the
same normal-only subsets. These are non-DP (they are not gradient-trained); the point is
to check that 'which samples you keep does not matter' is a property of the selection,
not of the autoencoder. Real outputs -> results/alt_detector.json. NEVER fabricate.
"""
import sys, json, logging
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.ERROR)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from experiments.cached_load import cached_load
from experiments.run_full_paper_sweep import DS_CFG
from models.siamese import SiameseEncoder, train_siamese
from experiments.principled_method import seq_temporal_dedup


def fps(E, b):
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    n = len(E); b = min(b, n); sel = [0]; d = 1.0 - E @ E[0]
    for _ in range(1, b):
        i = int(np.argmax(d)); sel.append(i); d = np.minimum(d, 1.0 - E @ E[i])
    return np.array(sorted(sel))


def score_f1(detector_name, Xtr, Xte, yte, pct, seed):
    if detector_name == "iforest":
        m = IsolationForest(n_estimators=100, random_state=seed).fit(Xtr)
        s = -m.score_samples(Xte)               # higher = more anomalous
    else:
        m = OneClassSVM(nu=0.1, gamma="scale").fit(Xtr)
        s = -m.decision_function(Xte)
    thr = np.percentile(s, pct)
    return f1_score(yte, (s >= thr).astype(int), zero_division=0)


def main():
    out = []
    for ds in ["skab", "swat"]:
        cfg = DS_CFG[ds]; pct = cfg["eval_percentile"]
        X, y = cached_load(ds, cfg["max_samples"])
        idx = np.arange(len(X)); tr, te = train_test_split(idx, test_size=0.2, random_state=0, stratify=y)
        tr = np.sort(tr); Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        enc = SiameseEncoder(Xtr.shape[1], cfg["siamese_emb"])
        enc = train_siamese(enc, Xtr, ytr, epochs=min(cfg["siamese_epochs"], 6), window=cfg["siamese_window"])
        Xn = Xtr[ytr == 0]
        if len(Xn) > 6000: Xn = Xn[:6000]
        E = enc.encode(Xn)
        kept, _ = seq_temporal_dedup(Xn, enc, keep_quantile=0.5); Xd = Xn[kept]; bud = len(Xd)
        Xf = Xn[fps(E, bud)]
        print(f"{ds}: normal={len(Xn)} budget={bud}", flush=True)
        for det in ["iforest", "ocsvm"]:
            for seed in range(5):
                rng = np.random.default_rng(1000 + seed)
                Xr = Xn[np.sort(rng.choice(len(Xn), bud, replace=False))]
                for mode, Xt in [("full", Xn), ("random", Xr), ("tdedup", Xd), ("fps", Xf)]:
                    f1 = score_f1(det, Xt, Xte, yte, pct, seed)
                    out.append({"dataset": ds, "detector": det, "mode": mode, "seed": seed,
                                "f1": round(float(f1), 4), "n": len(Xt)})
                    (ROOT / "results" / "alt_detector.json").write_text(json.dumps(out, indent=2))
                print(f"  {ds} {det} seed={seed} done", flush=True)
    print("\n=== ALT-DETECTOR SUMMARY (F1 mean over 5 seeds) ===")
    for ds in ["skab", "swat"]:
        for det in ["iforest", "ocsvm"]:
            r = {m: np.mean([x["f1"] for x in out if x["dataset"]==ds and x["detector"]==det and x["mode"]==m]) for m in ["full","random","tdedup","fps"]}
            sp = max(r.values()) - min(r.values())
            print(f"{ds} {det}: " + " ".join(f"{m}={r[m]:.3f}" for m in r) + f" | spread={sp:.3f}")


if __name__ == "__main__":
    main()
