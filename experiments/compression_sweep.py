r"""Compression-ratio sweep (reviewer M2): does the selection null hold at more
aggressive compression than the ~50% operating point of the main comparison?
For SKAB and SWaT, at eps=2 (centralised DP, normal-only), vary the temporal-dedup
keep-quantile to obtain several retention levels rho; at each level compare temporal
dedup against a matched-size random subsample, a matched-size diversity coreset (fps),
and full data. Same encoder/DP/eval machinery as characterization.py.
Output -> results/compression_sweep.json (+ prints a summary), incremental/resumable.
"""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.stdout.reconfigure(encoding="utf-8")
from sklearn.model_selection import train_test_split
from models.siamese import SiameseEncoder, train_siamese
from experiments.run_full_paper_sweep import DS_CFG
from experiments.principled_method import seq_temporal_dedup
from experiments.characterization import fps_coreset, dp_eval, _load

EPS = 2.0
SEEDS = list(range(5))
QUANTILES = [0.25, 0.5, 0.75]     # stricter -> lighter dedup; gives a rho spread
DATASETS = sys.argv[1:] if len(sys.argv) > 1 else ["skab", "swat"]
OUTP = ROOT / "results" / "compression_sweep.json"
done = json.load(open(OUTP)) if OUTP.exists() else []
seen = {(r["dataset"], r["q"], r["mode"], r["seed"]) for r in done}
out = list(done)

def main():
    for ds in DATASETS:
        cfg = dict(DS_CFG[ds]); cfg["siamese_epochs"] = min(cfg["siamese_epochs"], 6)
        X, y = _load(ds, cfg)
        idx = np.arange(len(X)); tr, te = train_test_split(idx, test_size=0.2, random_state=0, stratify=y); tr = np.sort(tr)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        enc = SiameseEncoder(Xtr.shape[1], cfg["siamese_emb"])
        enc = train_siamese(enc, Xtr, ytr, epochs=cfg["siamese_epochs"], window=cfg["siamese_window"])
        Xn = Xtr[ytr == 0]
        if len(Xn) > 6000: Xn = Xn[:6000]
        E = enc.encode(Xn)
        # full data (rho=1) once per seed, recorded under q=None
        for seed in SEEDS:
            if (ds, None, "full", seed) not in seen:
                f1, ap = dp_eval(Xn, Xte, yte, cfg, EPS, seed)
                out.append({"dataset": ds, "q": None, "rho": 1.0, "mode": "full", "seed": seed,
                            "f1": round(f1, 4), "n": len(Xn)})
                OUTP.write_text(json.dumps(out, indent=2))
                print(f"  {ds} full            seed={seed}: F1={f1:.4f}", flush=True)
        for q in QUANTILES:
            kept_t, _ = seq_temporal_dedup(Xn, enc, keep_quantile=q)
            Xd = Xn[kept_t]; budget = len(Xd); rho = budget / len(Xn)
            Xf = Xn[fps_coreset(E, budget)]
            print(f"[{ds}] q={q}: budget={budget} (rho={rho:.3f})", flush=True)
            for seed in SEEDS:
                rng = np.random.default_rng(2000 + seed)
                Xr = Xn[np.sort(rng.choice(len(Xn), size=budget, replace=False))]
                for mode, Xt in [("random", Xr), ("temporal", Xd), ("diversity", Xf)]:
                    if (ds, q, mode, seed) in seen:
                        continue
                    f1, ap = dp_eval(Xt, Xte, yte, cfg, EPS, seed)
                    out.append({"dataset": ds, "q": q, "rho": round(rho, 4), "mode": mode,
                                "seed": seed, "f1": round(f1, 4), "n": len(Xt)})
                    OUTP.write_text(json.dumps(out, indent=2))
                    print(f"  {ds} {mode:10s} q={q} rho={rho:.3f} seed={seed}: F1={f1:.4f}", flush=True)
    # summary
    print("\n=== COMPRESSION SWEEP SUMMARY (F1 mean+/-std, eps=2) ===")
    for ds in sorted(set(r["dataset"] for r in out)):
        fullv = [r["f1"] for r in out if r["dataset"] == ds and r["mode"] == "full"]
        if fullv:
            print(f"{ds} full (rho=1.00): {np.mean(fullv):.3f}+/-{np.std(fullv):.3f}")
        for q in QUANTILES:
            rows = [r for r in out if r["dataset"] == ds and r["q"] == q]
            if not rows: continue
            rho = rows[0]["rho"]; parts = []
            for mode in ["random", "temporal", "diversity"]:
                v = [r["f1"] for r in rows if r["mode"] == mode]
                parts.append(f"{mode}={np.mean(v):.3f}+/-{np.std(v):.3f}" if v else f"{mode}=NA")
            print(f"{ds} rho={rho:.2f}: " + "  ".join(parts))

if __name__ == "__main__":
    main()
