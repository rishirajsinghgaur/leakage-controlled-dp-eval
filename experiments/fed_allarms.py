r"""
DECISIVE federated experiment (reviewer: the federated 11/16 must be CONTROLLED with a
matched random arm, not argued away). Runs the selection comparison IN THE FEDERATED
setting with all four arms at matched per-client budget:
  full / random / tdedup / fps  (selection_mode in fl/client.py)
Contiguous + contiguous-heterogeneous partitions, end-to-end DP. Reuses the same
validated FedAvg+DP machinery as federated_sweep.py.

KEY QUESTION: in FL, does random tie dedup (null holds in the titular setting), or not?
Real outputs -> results/fed_allarms.json. NEVER fabricate.
"""
import sys, json, logging, argparse
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.ERROR)
from models.siamese import SiameseEncoder, train_siamese
from fl.client import build_client_fn
from fl.server import run_simulation
from experiments.run_full_paper_sweep import DS_CFG, K
from experiments.federated_sweep import contiguous_partitions
from experiments.cached_load import cached_load
from sklearn.model_selection import train_test_split


def run(datasets, seeds, epsilons, partitions):
    out = []
    done = set()
    op = ROOT / "results" / "fed_allarms.json"
    if op.exists():
        out = json.loads(op.read_text())
        done = {(r["dataset"], r["partition"], r["mode"], r["epsilon_target"], r["seed"]) for r in out}
    for ds in datasets:
        cfg = dict(DS_CFG[ds]); cfg["train_normal_only"] = True
        X, y = cached_load(ds, cfg["max_samples"])
        idx = np.arange(len(X)); tr, te = train_test_split(idx, test_size=0.2, random_state=0, stratify=y)
        tr = np.sort(tr); Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        enc = SiameseEncoder(Xtr.shape[1], cfg["siamese_emb"])
        enc = train_siamese(enc, Xtr, ytr, epochs=cfg["siamese_epochs"], window=cfg["siamese_window"])
        pub_normal = Xtr[ytr == 0]
        for part_label, het in partitions:
            for seed in seeds:
                np.random.seed(seed); import torch; torch.manual_seed(seed)
                parts = contiguous_partitions(Xtr, ytr, K, hetero=het, seed=seed)
                for eps in epsilons:
                    for mode in ["full", "random", "tdedup", "fps"]:
                        if (ds, part_label, mode, eps, seed) in done:
                            continue
                        c = dict(cfg); c["selection_mode"] = mode
                        cfn = build_client_fn(partitions=parts, input_dim=Xtr.shape[1], config=c,
                                              dedup_enabled=False, dp_enabled=True, target_epsilon=eps,
                                              siamese_encoder=enc, global_train_X=pub_normal)
                        r = run_simulation(client_fn=cfn, n_clients=K, n_rounds=cfg["n_rounds"],
                                           input_dim=Xtr.shape[1], config=c, X_test=Xte, y_test=yte,
                                           condition_name=mode, dataset_name=ds, seed=seed)
                        out.append({"dataset": ds, "partition": part_label, "mode": mode,
                                    "epsilon_target": eps, "seed": seed, "f1": r.final_f1,
                                    "auprc": r.final_auprc, "rho": r.final_rho})
                        op.write_text(json.dumps(out, indent=2))
                        print(f"{ds} {part_label} {mode} eps={eps} s={seed} F1={r.final_f1:.4f} rho={r.final_rho:.3f}", flush=True)
    print("\n=== FED ALL-ARMS SUMMARY (contig, F1 mean) ===")
    for ds in datasets:
        for eps in epsilons:
            v = {m: np.mean([r["f1"] for r in out if r["dataset"]==ds and r["partition"]=="contig" and r["mode"]==m and r["epsilon_target"]==eps] or [float('nan')]) for m in ["full","random","tdedup","fps"]}
            print(f"{ds} eps={eps}: " + " ".join(f"{m}={v[m]:.3f}" for m in v))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["skab", "swat"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epsilons", nargs="+", type=float, default=[0.5, 2.0])
    ap.add_argument("--partitions", default="contig")  # contig | both
    a = ap.parse_args()
    parts = [("contig", False)] if a.partitions == "contig" else [("contig", False), ("contig-het", True)]
    run(a.datasets, list(range(a.seeds)), a.epsilons, parts)
