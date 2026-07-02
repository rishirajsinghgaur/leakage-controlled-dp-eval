r"""
Novel-method test: DIVERSITY-PRESERVING coreset of normal states vs full normal vs
temporal dedup, for a DP reconstruction anomaly detector. Hypothesis: the FL benefit
washed out because per-client dedup STARVES normal-state diversity; a coreset that
explicitly MAXIMISES coverage (farthest-point sampling in encoder-embedding space)
should retain diverse normal states with a small set -> better detector + fewer DP steps.

Centralized fast diagnostic (capped, reduced passes) on SKAB+SWaT.
Modes: normal_full | normal_tdedup | normal_fps (k-center / farthest-point, same budget as tdedup).
Real outputs -> results/diversity_coreset.json.
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
PASSES = 15


def fps_coreset(E, budget):
    """Farthest-point sampling on L2-normalised embeddings E (cosine). Greedy k-center:
    maximises minimum pairwise distance -> diverse coverage of normal states."""
    n = len(E); budget = min(budget, n)
    sel = [0]
    d = 1.0 - E @ E[0]                       # cosine distance to first
    for _ in range(1, budget):
        i = int(np.argmax(d)); sel.append(i)
        d = np.minimum(d, 1.0 - E @ E[i])
    return np.array(sorted(sel))


def dp_eval(Xt, Xte, yte, cfg, eps, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(Xt); B = min(cfg["batch_size"], max(2, n - 1)); steps = max(1, n // B) * PASSES
    sig = compute_sigma_for_total_epsilon(eps, n, B, steps, 1, cfg["delta"])
    from opacus import PrivacyEngine
    m = AnomalyAutoencoder(Xt.shape[1], cfg["bottleneck"]); opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
    ld = DataLoader(TensorDataset(torch.from_numpy(Xt).float()), batch_size=B, shuffle=True, drop_last=True)
    pe = PrivacyEngine(); m, opt, ld = pe.make_private(module=m, optimizer=opt, data_loader=ld, noise_multiplier=sig, max_grad_norm=cfg["max_grad_norm"])
    lf = nn.MSELoss(); m.train()
    for _ in range(PASSES):
        for (b,) in ld:
            opt.zero_grad(); l = lf(m(b), b); l.backward(); opt.step()
    mm = getattr(m, "_module", m); met = evaluate_anomaly_detector(mm, Xte, yte, percentile=cfg["eval_percentile"])
    return met["f1"], met["auprc"]


def main():
    out = []
    for ds in ["skab", "swat"]:
        cfg = DS_CFG[ds]
        X, y, _ = DATASET_REGISTRY[ds](max_samples=cfg["max_samples"], random_state=42)
        idx = np.arange(len(X)); tr, te = train_test_split(idx, test_size=0.2, random_state=0, stratify=y); tr = np.sort(tr)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        enc = SiameseEncoder(Xtr.shape[1], cfg["siamese_emb"]); enc = train_siamese(enc, Xtr, ytr, epochs=cfg["siamese_epochs"], window=cfg["siamese_window"])
        Xn = Xtr[ytr == 0]
        if len(Xn) > 6000: Xn = Xn[:6000]
        E = enc.encode(Xn)
        kept_t, tau = seq_temporal_dedup(Xn, enc, keep_quantile=0.5); Xd = Xn[kept_t]
        kept_f = fps_coreset(E, budget=len(Xd)); Xf = Xn[kept_f]      # same budget as tdedup
        print(f"{ds}: normal={len(Xn)} tdedup={len(Xd)} fps={len(Xf)} (budget matched)", flush=True)
        for seed in (0, 1, 2):
            for eps in (0.5, 2.0):
                for mode, Xt in [("normal_full", Xn), ("normal_tdedup", Xd), ("normal_fps", Xf)]:
                    f1, ap = dp_eval(Xt, Xte, yte, cfg, eps, seed)
                    out.append({"dataset": ds, "mode": mode, "epsilon": eps, "seed": seed, "f1": round(f1, 4), "auprc": round(ap, 4), "n": len(Xt)})
                    print(f"  {ds} {mode} eps={eps} s={seed} F1={f1:.4f} AUPRC={ap:.4f}", flush=True)
                    (ROOT / "results" / "diversity_coreset.json").write_text(json.dumps(out, indent=2))
    print("\n=== DIVERSITY CORESET SUMMARY (F1 / AUPRC mean) ===")
    for ds in ["skab", "swat"]:
        for eps in (0.5, 2.0):
            def g(mode, k):
                v=[x[k] for x in out if x["dataset"]==ds and x["mode"]==mode and x["epsilon"]==eps]; return np.mean(v) if v else float('nan')
            print(f"{ds} eps={eps}: full F1={g('normal_full','f1'):.3f}/AP={g('normal_full','auprc'):.3f}  "
                  f"tdedup F1={g('normal_tdedup','f1'):.3f}/AP={g('normal_tdedup','auprc'):.3f}  "
                  f"FPS F1={g('normal_fps','f1'):.3f}/AP={g('normal_fps','auprc'):.3f}")


if __name__ == "__main__":
    main()
