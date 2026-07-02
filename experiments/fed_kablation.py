r"""
K-ablation (reviewer: K=5 is a toy federated setup; does the null survive larger fleets?).
Runs the federated all-arms comparison (full/random/tdedup/fps) at K in {10, 20}
contiguous clients, SWaT+SKAB, eps {0.5, 2.0}, 5 seeds. Resumable.
Output -> results/fed_kablation.json. NEVER fabricate.
"""
import sys, json, logging
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.ERROR)
from models.siamese import SiameseEncoder, train_siamese
from fl.client import build_client_fn
from fl.server import run_simulation
from experiments.run_full_paper_sweep import DS_CFG
from experiments.federated_sweep import contiguous_partitions
from experiments.cached_load import cached_load
from sklearn.model_selection import train_test_split

KS = [10, 20]; SEEDS = 5; EPS = [0.5, 2.0]
op = ROOT / "results" / "fed_kablation.json"
out = json.loads(op.read_text()) if op.exists() else []
done = {(r["dataset"], r["K"], r["mode"], r["epsilon_target"], r["seed"]) for r in out}

for ds in ["skab", "swat"]:
    cfg = dict(DS_CFG[ds]); cfg["train_normal_only"] = True
    X, y = cached_load(ds, cfg["max_samples"])
    idx = np.arange(len(X)); tr, te = train_test_split(idx, test_size=0.2, random_state=0, stratify=y)
    tr = np.sort(tr); Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    enc = SiameseEncoder(Xtr.shape[1], cfg["siamese_emb"])
    enc = train_siamese(enc, Xtr, ytr, epochs=cfg["siamese_epochs"], window=cfg["siamese_window"])
    pub_normal = Xtr[ytr == 0]
    for K in KS:
        for seed in range(SEEDS):
            np.random.seed(seed); import torch; torch.manual_seed(seed)
            parts = contiguous_partitions(Xtr, ytr, K, hetero=False, seed=seed)
            for eps in EPS:
                for mode in ["full", "random", "tdedup", "fps"]:
                    if (ds, K, mode, eps, seed) in done: continue
                    c = dict(cfg); c["selection_mode"] = mode
                    cfn = build_client_fn(partitions=parts, input_dim=Xtr.shape[1], config=c,
                                          dedup_enabled=False, dp_enabled=True, target_epsilon=eps,
                                          siamese_encoder=enc, global_train_X=pub_normal)
                    r = run_simulation(client_fn=cfn, n_clients=K, n_rounds=cfg["n_rounds"],
                                       input_dim=Xtr.shape[1], config=c, X_test=Xte, y_test=yte,
                                       condition_name=mode, dataset_name=ds, seed=seed)
                    out.append({"dataset": ds, "K": K, "mode": mode, "epsilon_target": eps,
                                "seed": seed, "f1": r.final_f1, "rho": r.final_rho})
                    op.write_text(json.dumps(out, indent=2))
                    print(f"{ds} K={K} {mode} eps={eps} s={seed} F1={r.final_f1:.4f}", flush=True)
print("\n=== K-ABLATION SUMMARY (F1 mean) ===")
for ds in ["skab", "swat"]:
    for K in KS:
        for eps in EPS:
            v = {m: np.mean([r["f1"] for r in out if r["dataset"]==ds and r["K"]==K and r["mode"]==m and r["epsilon_target"]==eps] or [float('nan')]) for m in ["full","random","tdedup","fps"]}
            print(f"{ds} K={K} eps={eps}: " + " ".join(f"{m}={v[m]:.3f}" for m in v))
