r"""
Factorial protocol ablation (reviewer top ask): WHICH protocol rule kills the utility
artifact? 2x2 over Rule1 (selection gate fit on PUBLIC normal vs PRIVATE full data) x
Rule2 (detector trained on NORMAL-only vs CONTAMINATED data). For each config we run
no-dedup (C5) and dedup (C6) on SWaT and report the deduplication 'gain' = F1(dedup) -
F1(nodedup). The uncontrolled config (private gate + contaminated) should show the spurious gain; fixing
the responsible rule(s) should remove it. Resumable.
Output -> results/protocol_ablation.json. NEVER fabricate.
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

DS = "swat"; EPS = [0.5, 2.0]; SEEDS = 5
# (label, gate_public, train_normal_only)
CONFIGS = [
    ("uncontrolled (R1 off, R2 off)",      False, False),
    ("fix R1 only (public gate)",   True,  False),
    ("fix R2 only (normal-only)",   False, True),
    ("controlled (R1+R2)",              True,  True),
]
op = ROOT / "results" / "protocol_ablation.json"
out = json.loads(op.read_text()) if op.exists() else []
done = {(r["config"], r["epsilon"], r["seed"], r["condition"]) for r in out}

cfg0 = dict(DS_CFG[DS])
X, y = cached_load(DS, cfg0["max_samples"])
idx = np.arange(len(X)); tr, te = train_test_split(idx, test_size=0.2, random_state=0, stratify=y)
tr = np.sort(tr); Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
enc = SiameseEncoder(Xtr.shape[1], cfg0["siamese_emb"]); enc = train_siamese(enc, Xtr, ytr, epochs=cfg0["siamese_epochs"], window=cfg0["siamese_window"])
pub_normal = Xtr[ytr == 0]          # Rule 1 ON: public normal-only gate reference
priv_full = Xtr                      # Rule 1 OFF: full private data (incl. anomalies)

for label, gate_pub, normal_only in CONFIGS:
    for seed in range(SEEDS):
        np.random.seed(seed); import torch; torch.manual_seed(seed)
        parts = partition_iid(Xtr, ytr, K, random_state=seed)
        gate_ref = pub_normal if gate_pub else priv_full
        for eps in EPS:
            for cond, dedup in [("C5_nodedup", False), ("C6_dedup", True)]:
                if (label, eps, seed, cond) in done: continue
                c = dict(cfg0); c["train_normal_only"] = normal_only
                cfn = build_client_fn(partitions=parts, input_dim=Xtr.shape[1], config=c,
                                      dedup_enabled=dedup, dp_enabled=True, target_epsilon=eps,
                                      siamese_encoder=enc if dedup else None, global_train_X=gate_ref)
                r = run_simulation(client_fn=cfn, n_clients=K, n_rounds=cfg0["n_rounds"],
                                   input_dim=Xtr.shape[1], config=c, X_test=Xte, y_test=yte,
                                   condition_name=cond, dataset_name=DS, seed=seed)
                out.append({"config": label, "gate_public": gate_pub, "normal_only": normal_only,
                            "epsilon": eps, "seed": seed, "condition": cond, "f1": r.final_f1})
                op.write_text(json.dumps(out, indent=2))
                print(f"{label} eps={eps} s={seed} {cond} F1={r.final_f1:.4f}", flush=True)

print("\n=== PROTOCOL ABLATION: dedup gain = F1(dedup) - F1(nodedup) ===")
for label, gp, no in CONFIGS:
    for eps in EPS:
        c5 = np.mean([r["f1"] for r in out if r["config"]==label and r["epsilon"]==eps and r["condition"]=="C5_nodedup"] or [np.nan])
        c6 = np.mean([r["f1"] for r in out if r["config"]==label and r["epsilon"]==eps and r["condition"]=="C6_dedup"] or [np.nan])
        print(f"  {label:30s} eps={eps}: nodedup={c5:.3f} dedup={c6:.3f} gain={c6-c5:+.3f}")
