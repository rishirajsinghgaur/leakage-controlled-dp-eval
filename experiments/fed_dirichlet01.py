r"""
More-heterogeneous federated arm (reviewer item 12): repeat the four-arm selection
comparison under a Dirichlet(alpha=0.1) non-IID split (much more heterogeneous than the
alpha=0.5 used elsewhere), to check the null survives strong client heterogeneity.
SWaT+SKAB, eps {0.5,2.0}, K=5, 5 seeds. Resumable.
Output -> results/fed_dirichlet01.json. NEVER fabricate.
"""
import sys, json, logging
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.ERROR)
from data.partitioner import partition_dirichlet
from models.siamese import SiameseEncoder, train_siamese
from fl.client import build_client_fn
from fl.server import run_simulation
from experiments.run_full_paper_sweep import DS_CFG, K
from experiments.cached_load import cached_load
from sklearn.model_selection import train_test_split

ALPHA = 0.1; EPS = [0.5, 2.0]; SEEDS = 5
op = ROOT / "results" / "fed_dirichlet01.json"
out = json.loads(op.read_text()) if op.exists() else []
done = {(r["dataset"], r["mode"], r["epsilon_target"], r["seed"]) for r in out}

for ds in ["skab", "swat"]:
    cfg = dict(DS_CFG[ds]); cfg["train_normal_only"] = True
    X, y = cached_load(ds, cfg["max_samples"])
    idx = np.arange(len(X)); tr, te = train_test_split(idx, test_size=0.2, random_state=0, stratify=y)
    tr = np.sort(tr); Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    enc = SiameseEncoder(Xtr.shape[1], cfg["siamese_emb"])
    enc = train_siamese(enc, Xtr, ytr, epochs=cfg["siamese_epochs"], window=cfg["siamese_window"])
    pub_normal = Xtr[ytr == 0]
    for seed in range(SEEDS):
        np.random.seed(seed); import torch; torch.manual_seed(seed)
        parts = partition_dirichlet(Xtr, ytr, K, alpha=ALPHA, random_state=seed)
        for eps in EPS:
            for mode in ["full", "random", "tdedup", "fps"]:
                if (ds, mode, eps, seed) in done: continue
                c = dict(cfg); c["selection_mode"] = mode
                cfn = build_client_fn(partitions=parts, input_dim=Xtr.shape[1], config=c,
                                      dedup_enabled=False, dp_enabled=True, target_epsilon=eps,
                                      siamese_encoder=enc, global_train_X=pub_normal)
                r = run_simulation(client_fn=cfn, n_clients=K, n_rounds=cfg["n_rounds"],
                                   input_dim=Xtr.shape[1], config=c, X_test=Xte, y_test=yte,
                                   condition_name=mode, dataset_name=ds, seed=seed)
                out.append({"dataset": ds, "mode": mode, "epsilon_target": eps, "seed": seed,
                            "f1": r.final_f1, "rho": r.final_rho})
                op.write_text(json.dumps(out, indent=2))
                print(f"{ds} dir0.1 {mode} eps={eps} s={seed} F1={r.final_f1:.4f}", flush=True)
print("\n=== DIRICHLET(0.1) SUMMARY (F1 mean) ===")
for ds in ["skab", "swat"]:
    for eps in EPS:
        v = {m: np.mean([r["f1"] for r in out if r["dataset"]==ds and r["mode"]==m and r["epsilon_target"]==eps] or [float('nan')]) for m in ["full","random","tdedup","fps"]}
        print(f"{ds} eps={eps}: " + " ".join(f"{m}={v[m]:.3f}" for m in v))
