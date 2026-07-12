r"""Threshold-sensitivity sweep (reviewer Issue 10): F1 uses a fixed percentile threshold;
does the selection null depend on that choice? For each arm we train the DP detector once
(eps=2, 5 seeds) and evaluate point-wise F1 at several percentile thresholds. If no learned
rule beats random or full at ANY threshold, the null is threshold-robust. SKAB+SWaT.
Output -> results/threshold_sweep.json (+ printed summary). Resumable.
"""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.stdout.reconfigure(encoding="utf-8")
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from models.mlp import AnomalyAutoencoder, evaluate_anomaly_detector
from models.siamese import SiameseEncoder, train_siamese
from privacy.accountant import compute_sigma_for_total_epsilon
from experiments.run_full_paper_sweep import DS_CFG
from experiments.principled_method import seq_temporal_dedup
from experiments.characterization import fps_coreset, _load

EPS = 2.0; SEEDS = list(range(5)); PASSES = 15
PERC = [85, 88, 90, 92, 95]
DATASETS = sys.argv[1:] if len(sys.argv) > 1 else ["skab", "swat"]
OUTP = ROOT / "results" / "threshold_sweep.json"
done = json.load(open(OUTP)) if OUTP.exists() else []
seen = {(r["dataset"], r["mode"], r["seed"], r["percentile"]) for r in done}
out = list(done)


def train_dp(Xt, cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(Xt); B = min(cfg["batch_size"], max(2, n - 1)); steps = max(1, n // B) * PASSES
    sig = compute_sigma_for_total_epsilon(EPS, n, B, steps, 1, cfg["delta"])
    from opacus import PrivacyEngine
    m = AnomalyAutoencoder(Xt.shape[1], cfg["bottleneck"]); opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
    ld = DataLoader(TensorDataset(torch.from_numpy(Xt).float()), batch_size=B, shuffle=True, drop_last=True)
    m, opt, ld = PrivacyEngine().make_private(module=m, optimizer=opt, data_loader=ld, noise_multiplier=sig, max_grad_norm=cfg["max_grad_norm"])
    lf = nn.MSELoss(); m.train()
    for _ in range(PASSES):
        for (b,) in ld:
            opt.zero_grad(); lf(m(b), b).backward(); opt.step()
    return getattr(m, "_module", m)


def main():
    for ds in DATASETS:
        cfg = dict(DS_CFG[ds]); cfg["siamese_epochs"] = min(cfg["siamese_epochs"], 6)
        X, y = _load(ds, cfg)
        idx = np.arange(len(X)); tr, te = train_test_split(idx, test_size=0.2, random_state=0, stratify=y); tr = np.sort(tr)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        enc = SiameseEncoder(Xtr.shape[1], cfg["siamese_emb"]); enc = train_siamese(enc, Xtr, ytr, epochs=cfg["siamese_epochs"], window=cfg["siamese_window"])
        Xn = Xtr[ytr == 0]
        if len(Xn) > 6000: Xn = Xn[:6000]
        E = enc.encode(Xn)
        kept, _ = seq_temporal_dedup(Xn, enc, keep_quantile=0.5); budget = len(kept)
        Xd = Xn[kept]; Xf = Xn[fps_coreset(E, budget)]
        for seed in SEEDS:
            rng = np.random.default_rng(4000 + seed)
            Xr = Xn[np.sort(rng.choice(len(Xn), size=budget, replace=False))]
            for mode, Xt in [("full", Xn), ("random", Xr), ("temporal", Xd), ("diversity", Xf)]:
                if all((ds, mode, seed, p) in seen for p in PERC):
                    continue
                model = train_dp(Xt, cfg, seed)
                for p in PERC:
                    if (ds, mode, seed, p) in seen: continue
                    met = evaluate_anomaly_detector(model, Xte, yte, percentile=float(p))
                    out.append({"dataset": ds, "mode": mode, "seed": seed, "percentile": p, "f1": round(met["f1"], 4)})
                OUTP.write_text(json.dumps(out, indent=2))
                print(f"  {ds} {mode:10s} seed={seed}: " + " ".join(f"p{p}={[o['f1'] for o in out if o['dataset']==ds and o['mode']==mode and o['seed']==seed and o['percentile']==p][0]:.3f}" for p in PERC), flush=True)

    print("\n=== THRESHOLD SWEEP SUMMARY (F1 mean+/-std, eps=2, 5 seeds) ===")
    for ds in sorted(set(r["dataset"] for r in out)):
        print(f"[{ds}]  percentiles {PERC}")
        for m in ["full", "random", "temporal", "diversity"]:
            row = []
            for p in PERC:
                v = [r["f1"] for r in out if r["dataset"] == ds and r["mode"] == m and r["percentile"] == p]
                row.append(f"{np.mean(v):.3f}" if v else "NA")
            print(f"  {m:10s}: " + "  ".join(row))


if __name__ == "__main__":
    main()
