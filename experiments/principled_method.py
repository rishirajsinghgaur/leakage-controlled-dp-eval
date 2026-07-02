r"""
Genuine method-fix attempt (user chose 'attempt a fix first').
Diagnose+test a PRINCIPLED redesign on SKAB+SWaT under DP, centralized (isolates the
detector/dedup effect before re-adding FL):

  mixed_full   : reconstruction AE trained on ALL training data (current C5 behaviour)
  normal_full  : AE trained on NORMAL-only (y=0) data — the correct setup for a
                 reconstruction anomaly detector
  normal_tdedup: AE trained on temporally-deduplicated normal data — SEQUENTIAL dedup
                 in time order with an ADAPTIVE threshold: keep a sample only if its
                 embedding cosine-sim to the last KEPT sample < tau_adaptive
                 (tau from the consecutive-similarity distribution). Removes consecutive
                 redundancy, retains diverse normal states (no global collapse).

If normal_tdedup >= normal_full robustly under DP, the principled method genuinely helps;
else fall back to the characterization. Real outputs -> results/principled_method.json.
"""
import sys, json, logging
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.ERROR)
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from data.loaders import DATASET_REGISTRY
from models.siamese import SiameseEncoder, train_siamese
from models.mlp import AnomalyAutoencoder, evaluate_anomaly_detector
from privacy.accountant import compute_sigma_for_total_epsilon
from experiments.run_full_paper_sweep import DS_CFG


def seq_temporal_dedup(Xn, enc, keep_quantile=0.5):
    """Sequential temporal dedup of a time-ordered normal stream.
    Keep sample i iff cos-sim(emb_i, emb_lastkept) < tau, tau = quantile of consecutive sims.
    Returns kept indices. Adaptive: tau from the data's own consecutive-similarity distn."""
    E = enc.encode(Xn)                       # already L2-normalised
    cons = np.sum(E[1:] * E[:-1], axis=1)    # consecutive cosine sims
    tau = float(np.quantile(cons, keep_quantile))   # adaptive threshold
    kept = [0]; last = E[0]
    for i in range(1, len(E)):
        if float(np.dot(E[i], last)) < tau:
            kept.append(i); last = E[i]
    return np.array(kept), tau


def dp_train_eval(Xtrain, Xte, yte, cfg, eps, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(Xtrain); B = min(cfg["batch_size"], max(2, n - 1))
    steps = max(1, n // B) * cfg["local_epochs"]
    sigma = compute_sigma_for_total_epsilon(eps, n, B, steps, cfg["n_rounds"], cfg["delta"])
    model = AnomalyAutoencoder(Xtrain.shape[1], cfg["bottleneck"])
    from opacus import PrivacyEngine
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtrain).float()), batch_size=B, shuffle=True, drop_last=True)
    pe = PrivacyEngine()
    model, opt, loader = pe.make_private(module=model, optimizer=opt, data_loader=loader,
                                         noise_multiplier=sigma, max_grad_norm=cfg["max_grad_norm"])
    lossf = nn.MSELoss(); model.train()
    for _ in range(cfg["local_epochs"] * cfg["n_rounds"]):
        for (b,) in loader:
            opt.zero_grad(); loss = lossf(model(b), b); loss.backward(); opt.step()
    m = getattr(model, "_module", model)
    met = evaluate_anomaly_detector(m, Xte, yte, percentile=cfg["eval_percentile"])
    return met["f1"], met["auprc"], n


def run(datasets=("skab", "swat"), seeds=(0, 1, 2), epsilons=(0.5, 1.0, 2.0)):
    out = []
    for ds in datasets:
        cfg = DS_CFG[ds]
        X, y, _ = DATASET_REGISTRY[ds](max_samples=cfg["max_samples"], random_state=42)
        # keep temporal order in train; stratified test split via index sort
        idx = np.arange(len(X)); tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=0, stratify=y)
        tr_idx = np.sort(tr_idx)  # preserve temporal order in training stream
        Xtr, ytr, Xte, yte = X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]
        enc = SiameseEncoder(Xtr.shape[1], cfg["siamese_emb"])
        enc = train_siamese(enc, Xtr, ytr, epochs=cfg["siamese_epochs"], window=cfg["siamese_window"])
        Xn = Xtr[ytr == 0]                    # normal-only, time-ordered
        kept, tau = seq_temporal_dedup(Xn, enc, keep_quantile=0.5)
        Xn_dedup = Xn[kept]
        print(f"{ds}: normal={len(Xn)} -> tdedup={len(Xn_dedup)} (rho={len(Xn_dedup)/len(Xn):.3f}, tau={tau:.4f})", flush=True)
        for seed in seeds:
            for eps in epsilons:
                for mode, Xt in [("mixed_full", Xtr), ("normal_full", Xn), ("normal_tdedup", Xn_dedup)]:
                    f1, ap, n = dp_train_eval(Xt, Xte, yte, cfg, eps, seed)
                    out.append({"dataset": ds, "mode": mode, "epsilon": eps, "seed": seed,
                                "f1": round(f1, 4), "auprc": round(ap, 4), "n": int(n)})
                    print(f"  {ds} {mode} eps={eps} s={seed} F1={f1:.4f} AUPRC={ap:.4f} n={n}", flush=True)
                    (ROOT / "results" / "principled_method.json").write_text(json.dumps(out, indent=2))
    print("\n=== PRINCIPLED METHOD SUMMARY (F1 mean) ===")
    for ds in datasets:
        for eps in epsilons:
            r = {m: np.mean([x["f1"] for x in out if x["dataset"] == ds and x["mode"] == m and x["epsilon"] == eps])
                 for m in ["mixed_full", "normal_full", "normal_tdedup"]}
            print(f"{ds} eps={eps}: mixed={r['mixed_full']:.3f} normal_full={r['normal_full']:.3f} normal_tdedup={r['normal_tdedup']:.3f}")


if __name__ == "__main__":
    run()
