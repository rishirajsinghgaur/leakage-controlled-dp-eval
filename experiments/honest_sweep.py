r"""
Sweep for the corrected method. Real outputs -> results/honest_sweep.json.
Method: DP-FL reconstruction anomaly detector trained on NORMAL-ONLY data
  C5h = normal-only, full          (corrected baseline)
  C6h = normal-only + temporal dedup (adaptive sequential, efficiency + small utility)
Partitions are TIME-CONTIGUOUS blocks (each client = a contiguous segment of the stream,
realistic for IIoT edge nodes and required for temporal dedup to be meaningful):
  'contig'    = equal contiguous blocks
  'contig-het'= unequal contiguous blocks (heterogeneous, non-IID)
SKAB+SWaT+TEP, eps {0.5,1,2,4}. Public-normal SIL fit (already fixed in client).
"""
import sys, json, logging, argparse
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.ERROR)
from data.loaders import DATASET_REGISTRY
from data.partitioner import ClientPartition
from models.siamese import SiameseEncoder, train_siamese
from fl.client import build_client_fn
from fl.server import run_simulation
from experiments.run_full_paper_sweep import DS_CFG, K
from sklearn.model_selection import train_test_split


def contiguous_partitions(X, y, k, hetero=False, seed=0):
    n = len(X);
    if not hetero:
        bnd = np.linspace(0, n, k + 1).astype(int)
    else:
        rng = np.random.default_rng(seed)
        w = rng.dirichlet(np.ones(k) * 0.5); bnd = np.concatenate([[0], np.cumsum((w * n).astype(int))]); bnd[-1] = n
    parts = []
    for i in range(k):
        sl = slice(bnd[i], bnd[i + 1])
        parts.append(ClientPartition(client_id=i, X=X[sl], y=y[sl]))
    return parts


def run(datasets, seeds, epsilons):
    out = []
    for ds in datasets:
        cfg = dict(DS_CFG[ds]); cfg["train_normal_only"] = True   # base flag; C6h adds temporal_dedup
        from experiments.cached_load import cached_load
        X, y = cached_load(ds, cfg["max_samples"])
        idx = np.arange(len(X)); tr, te = train_test_split(idx, test_size=0.2, random_state=0, stratify=y)
        tr = np.sort(tr); Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        enc = SiameseEncoder(Xtr.shape[1], cfg["siamese_emb"])
        enc = train_siamese(enc, Xtr, ytr, epochs=cfg["siamese_epochs"], window=cfg["siamese_window"])
        pub_normal = Xtr[ytr == 0]
        for part_label, het in [("contig", False), ("contig-het", True)]:
            for seed in seeds:
                np.random.seed(seed); import torch; torch.manual_seed(seed)
                parts = contiguous_partitions(Xtr, ytr, K, hetero=het, seed=seed)
                for eps in epsilons:
                    for cond, tdedup in [("C5h_normalfull", False), ("C6h_normaltdedup", True)]:
                        c = dict(cfg); c["temporal_dedup"] = tdedup
                        cfn = build_client_fn(partitions=parts, input_dim=Xtr.shape[1], config=c,
                                              dedup_enabled=False, dp_enabled=True, target_epsilon=eps,
                                              siamese_encoder=enc, global_train_X=pub_normal)
                        r = run_simulation(client_fn=cfn, n_clients=K, n_rounds=cfg["n_rounds"],
                                           input_dim=Xtr.shape[1], config=c, X_test=Xte, y_test=yte,
                                           condition_name=cond, dataset_name=ds, seed=seed)
                        out.append({"dataset": ds, "partition": part_label, "condition": cond,
                                    "epsilon_target": eps, "seed": seed, "f1": r.final_f1,
                                    "auprc": r.final_auprc, "rho": r.final_rho})
                        (ROOT / "results" / "honest_sweep.json").write_text(json.dumps(out, indent=2))
                        print(f"{ds} {part_label} {cond} eps={eps} s={seed} F1={r.final_f1:.4f} AUPRC={r.final_auprc:.4f} rho={r.final_rho:.3f}", flush=True)
    # summary
    print("\n=== SWEEP SUMMARY (contig, F1/AUPRC mean) ===")
    for ds in datasets:
        for eps in epsilons:
            def agg(cond, key):
                v=[r[key] for r in out if r["dataset"]==ds and r["partition"]=="contig" and r["condition"]==cond and r["epsilon_target"]==eps]
                return np.mean(v) if v else float('nan')
            print(f"{ds} eps={eps}: C5h F1={agg('C5h_normalfull','f1'):.3f}/AP={agg('C5h_normalfull','auprc'):.3f}  "
                  f"C6h F1={agg('C6h_normaltdedup','f1'):.3f}/AP={agg('C6h_normaltdedup','auprc'):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["skab", "swat", "tep"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epsilons", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    a = ap.parse_args()
    run(a.datasets, list(range(a.seeds)), a.epsilons)
