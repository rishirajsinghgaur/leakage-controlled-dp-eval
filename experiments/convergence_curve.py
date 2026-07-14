r"""
Convergence curves: record per-round test F1 and train loss for a representative
federated run on each dataset, to justify R=15 rounds as sufficient.
Output -> results/convergence_curve.json.
"""
import sys, json, logging
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.ERROR)
from data.partitioner import partition_iid
from models.siamese import SiameseEncoder, train_siamese
from fl.client import build_client_fn
from fl.server import run_simulation
from experiments.run_full_paper_sweep import DS_CFG, K
from experiments.cached_load import cached_load
from sklearn.model_selection import train_test_split

out = {}
for ds in ["skab", "swat"]:
    cfg = dict(DS_CFG[ds]); cfg["train_normal_only"] = True; cfg["selection_mode"] = "full"
    cfg["n_rounds"] = 25                      # run longer than 15 to show the plateau
    X, y = cached_load(ds, cfg["max_samples"])
    idx = np.arange(len(X)); tr, te = train_test_split(idx, test_size=0.2, random_state=0, stratify=y)
    tr = np.sort(tr); Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    enc = SiameseEncoder(Xtr.shape[1], cfg["siamese_emb"]); enc = train_siamese(enc, Xtr, ytr, epochs=cfg["siamese_epochs"], window=cfg["siamese_window"])
    pub = Xtr[ytr == 0]
    import torch; np.random.seed(0); torch.manual_seed(0)
    parts = partition_iid(Xtr, ytr, K, random_state=0)
    cfn = build_client_fn(partitions=parts, input_dim=Xtr.shape[1], config=cfg, dedup_enabled=False,
                          dp_enabled=True, target_epsilon=2.0, siamese_encoder=enc, global_train_X=pub)
    r = run_simulation(client_fn=cfn, n_clients=K, n_rounds=cfg["n_rounds"], input_dim=Xtr.shape[1],
                       config=cfg, X_test=Xte, y_test=yte, condition_name="full", dataset_name=ds, seed=0)
    out[ds] = [{"round": h["round"], "f1": h["f1"], "train_loss": h["train_loss"]} for h in r.history]
    print(f"{ds}: {len(out[ds])} rounds, F1@15={[h['f1'] for h in out[ds] if h['round']==15]}", flush=True)
(ROOT / "results" / "convergence_curve.json").write_text(json.dumps(out, indent=2))
print("wrote results/convergence_curve.json")
