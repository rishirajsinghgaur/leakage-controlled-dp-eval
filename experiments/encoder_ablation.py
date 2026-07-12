r"""Encoder-quality ablation (reviewer Issue 3): is the selection null an artifact of a
weak similarity encoder? We repeat the learned-rule comparison using TWO representations:
  (a) the trained triplet encoder (as in the paper), and
  (b) RAW L2-normalised features (no encoder at all).
If learned rules (temporal dedup, diversity coreset) fail to beat random under BOTH
representations, the null is not a property of the encoder. eps=2, 5 seeds, SKAB+SWaT.
Output -> results/encoder_ablation.json  (+ printed summary). Resumable.
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
DATASETS = sys.argv[1:] if len(sys.argv) > 1 else ["skab", "swat"]
OUTP = ROOT / "results" / "encoder_ablation.json"
done = json.load(open(OUTP)) if OUTP.exists() else []
seen = {(r["dataset"], r["rep"], r["mode"], r["seed"]) for r in done}
out = list(done)


class RawEncoder:
    """Identity 'encoder': L2-normalised raw features (cosine == raw cosine)."""
    def encode(self, X):
        X = np.asarray(X, dtype=np.float32)
        nrm = np.linalg.norm(X, axis=1, keepdims=True); nrm[nrm == 0] = 1.0
        return X / nrm


def main():
    for ds in DATASETS:
        cfg = dict(DS_CFG[ds]); cfg["siamese_epochs"] = min(cfg["siamese_epochs"], 6)
        X, y = _load(ds, cfg)
        idx = np.arange(len(X)); tr, te = train_test_split(idx, test_size=0.2, random_state=0, stratify=y); tr = np.sort(tr)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        enc = SiameseEncoder(Xtr.shape[1], cfg["siamese_emb"])
        enc = train_siamese(enc, Xtr, ytr, epochs=cfg["siamese_epochs"], window=cfg["siamese_window"])
        raw = RawEncoder()
        Xn = Xtr[ytr == 0]
        if len(Xn) > 6000: Xn = Xn[:6000]

        # full data once per seed (rep-independent)
        for seed in SEEDS:
            if (ds, "-", "full", seed) not in seen:
                f1, ap = dp_eval(Xn, Xte, yte, cfg, EPS, seed)
                out.append({"dataset": ds, "rep": "-", "mode": "full", "seed": seed, "rho": 1.0, "f1": round(f1, 4)})
                OUTP.write_text(json.dumps(out, indent=2)); print(f"  {ds} full seed={seed}: {f1:.4f}", flush=True)

        for rep, enc_obj in [("encoder", enc), ("raw", raw)]:
            kept, _ = seq_temporal_dedup(Xn, enc_obj, keep_quantile=0.5)
            budget = len(kept); rho = budget / len(Xn)
            E = enc_obj.encode(Xn)
            Xd = Xn[kept]                       # temporal dedup (this representation)
            Xf = Xn[fps_coreset(E, budget)]     # diversity coreset (this representation)
            print(f"[{ds}/{rep}] budget={budget} rho={rho:.3f}", flush=True)
            for seed in SEEDS:
                rng = np.random.default_rng(3000 + seed)
                Xr = Xn[np.sort(rng.choice(len(Xn), size=budget, replace=False))]
                for mode, Xt in [("random", Xr), ("temporal", Xd), ("diversity", Xf)]:
                    if (ds, rep, mode, seed) in seen:
                        continue
                    f1, ap = dp_eval(Xt, Xte, yte, cfg, EPS, seed)
                    out.append({"dataset": ds, "rep": rep, "mode": mode, "seed": seed, "rho": round(rho, 4), "f1": round(f1, 4)})
                    OUTP.write_text(json.dumps(out, indent=2))
                    print(f"  {ds} {rep} {mode:10s} seed={seed}: {f1:.4f}", flush=True)

    print("\n=== ENCODER ABLATION SUMMARY (F1 mean+/-std, eps=2, 5 seeds) ===")
    for ds in sorted(set(r["dataset"] for r in out)):
        fu = [r["f1"] for r in out if r["dataset"] == ds and r["mode"] == "full"]
        print(f"{ds} full: {np.mean(fu):.3f}+/-{np.std(fu):.3f}")
        for rep in ["encoder", "raw"]:
            parts = []
            for m in ["random", "temporal", "diversity"]:
                v = [r["f1"] for r in out if r["dataset"] == ds and r["rep"] == rep and r["mode"] == m]
                if v: parts.append(f"{m}={np.mean(v):.3f}+/-{np.std(v):.3f}")
            if parts: print(f"  {ds} [{rep}]: " + "  ".join(parts))


if __name__ == "__main__":
    main()
