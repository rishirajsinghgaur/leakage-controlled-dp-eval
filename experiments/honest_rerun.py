r"""
Re-run: C5 (FedAvg+DP, no dedup) vs C6 (DP-FL anomaly detection) with the corrected
SIL-gate fit (PUBLIC normal-only reference, y=0) — leakage-controlled and statistically
correct. Determines whether the method genuinely helps once the gate no longer uses
private/anomalous data. Real outputs only; writes results/honest_rerun.json.

SKAB + SWaT + TEP, IID + Dirichlet, eps {0.5,1,2,4}, seeds 0-4 (fast first read).
"""
import sys, json, logging
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.WARNING)

from data.loaders import DATASET_REGISTRY
from data.partitioner import partition_iid, partition_dirichlet
from models.siamese import SiameseEncoder, train_siamese
from fl.client import build_client_fn
from fl.server import run_simulation
from experiments.run_full_paper_sweep import DS_CFG, K
from sklearn.model_selection import train_test_split


def run(datasets=("skab", "swat", "tep"), seeds=(0, 1, 2, 3, 4), epsilons=(0.5, 1.0, 2.0, 4.0)):
    out = []
    for ds in datasets:
        cfg = DS_CFG[ds]
        X, y, _ = DATASET_REGISTRY[ds](max_samples=cfg["max_samples"], random_state=42)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
        enc = SiameseEncoder(Xtr.shape[1], cfg["siamese_emb"])
        enc = train_siamese(enc, Xtr, ytr, epochs=cfg["siamese_epochs"], window=cfg["siamese_window"])
        pub_normal = Xtr[ytr == 0]   # public normal-operation reference for the SIL gate
        for alpha, plabel in [(None, "iid"), (0.5, "dir0.5")]:
            for seed in seeds:
                np.random.seed(seed)
                import torch; torch.manual_seed(seed)
                parts = (partition_iid(Xtr, ytr, K, random_state=seed) if alpha is None
                         else partition_dirichlet(Xtr, ytr, K, alpha=alpha, random_state=seed))
                for eps in epsilons:
                    for cond, dedup in [("C5_nodedup", False), ("C6_dedup", True)]:
                        cfn = build_client_fn(
                            partitions=parts, input_dim=Xtr.shape[1], config=cfg,
                            dedup_enabled=dedup, dp_enabled=True, target_epsilon=eps,
                            siamese_encoder=enc if dedup else None,
                            global_train_X=pub_normal)
                        r = run_simulation(client_fn=cfn, n_clients=K, n_rounds=cfg["n_rounds"],
                                           input_dim=Xtr.shape[1], config=cfg, X_test=Xte, y_test=yte,
                                           condition_name=cond, dataset_name=ds, seed=seed)
                        out.append({"dataset": ds, "partition": plabel, "condition": cond,
                                    "epsilon_target": eps, "seed": seed, "f1": r.final_f1,
                                    "rho": r.final_rho})
                        (ROOT / "results" / "honest_rerun.json").write_text(json.dumps(out, indent=2))
                        print(f"{ds} {plabel} {cond} eps={eps} s={seed} F1={r.final_f1:.4f} rho={r.final_rho:.3f}", flush=True)
    print("\n=== RE-RUN SUMMARY (F1 mean over seeds) ===")
    for ds in datasets:
        for part in ["iid", "dir0.5"]:
            for eps in epsilons:
                c5 = [r["f1"] for r in out if r["dataset"]==ds and r["partition"]==part and r["condition"]=="C5_nodedup" and r["epsilon_target"]==eps]
                c6 = [r["f1"] for r in out if r["dataset"]==ds and r["partition"]==part and r["condition"]=="C6_dedup" and r["epsilon_target"]==eps]
                rho = np.mean([r["rho"] for r in out if r["dataset"]==ds and r["partition"]==part and r["condition"]=="C6_dedup" and r["epsilon_target"]==eps])
                if c5 and c6:
                    d = np.mean(c6)-np.mean(c5)
                    print(f"{ds} {part} eps={eps}: C5={np.mean(c5):.3f} C6={np.mean(c6):.3f} d={d:+.3f} rho={rho:.3f} {'WIN' if d>0.003 else ('LOSS' if d<-0.003 else 'tie')}")


if __name__ == "__main__":
    run()
