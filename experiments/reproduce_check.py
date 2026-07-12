r"""Reproducibility audit (read-only): re-run the CORE pipeline through the IDENTICAL code
path on a representative subset (SWaT+SKAB, seeds 0-1, all budgets, all arms) and compare to
the committed results/characterization.json. Does NOT overwrite anything; writes only to
reproduction_check/. Exact seeded reproduction => the whole pipeline is deterministic."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.stdout.reconfigure(encoding="utf-8")
import logging; logging.basicConfig(level=logging.ERROR)
from sklearn.model_selection import train_test_split
from models.siamese import SiameseEncoder, train_siamese
from experiments.characterization import dp_eval, fps_coreset, _load   # identical code path
from experiments.run_full_paper_sweep import DS_CFG
from experiments.principled_method import seq_temporal_dedup

SEEDS = [0, 1]; EPS = [0.5, 1.0, 2.0, 4.0]; DATASETS = ["swat", "skab"]
OUT = ROOT / "reproduction_check"; OUT.mkdir(exist_ok=True)
committed = {(r["dataset"], r["mode"], r["epsilon"], r["seed"]): r
             for r in json.load(open(ROOT/"results"/"characterization.json"))}

rows = []; TOL = 0.005
for ds in DATASETS:
    cfg = dict(DS_CFG[ds]); cfg["siamese_epochs"] = min(cfg["siamese_epochs"], 6)
    X, y = _load(ds, cfg)
    idx = np.arange(len(X)); tr, te = train_test_split(idx, test_size=0.2, random_state=0, stratify=y); tr = np.sort(tr)
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    enc = SiameseEncoder(Xtr.shape[1], cfg["siamese_emb"]); enc = train_siamese(enc, Xtr, ytr, epochs=cfg["siamese_epochs"], window=cfg["siamese_window"])
    Xn = Xtr[ytr == 0]
    if len(Xn) > 6000: Xn = Xn[:6000]
    E = enc.encode(Xn)
    kept_t, _ = seq_temporal_dedup(Xn, enc, keep_quantile=0.5); Xd = Xn[kept_t]; budget = len(Xd)
    Xf = Xn[fps_coreset(E, budget)]
    for seed in SEEDS:
        rng = np.random.default_rng(1000 + seed)
        Xr = Xn[np.sort(rng.choice(len(Xn), size=budget, replace=False))]
        for eps in EPS:
            for mode, Xt in [("full", Xn), ("random", Xr), ("tdedup", Xd), ("fps", Xf)]:
                f1, ap = dp_eval(Xt, Xte, yte, cfg, eps, seed)
                key = (ds, mode, eps, seed); old = committed.get(key)
                d = abs(f1 - old["f1"]) if old else None
                status = "PASS" if (old and d <= TOL) else ("FAIL" if old else "NO-BASELINE")
                rows.append({"dataset": ds, "mode": mode, "epsilon": eps, "seed": seed,
                             "f1_new": round(f1, 4), "f1_committed": old["f1"] if old else None,
                             "abs_diff": round(d, 4) if d is not None else None, "status": status})
                print(f"  {ds} {mode} eps={eps} s={seed}: new={f1:.4f} committed={old['f1'] if old else 'NA'} diff={d:.4f} [{status}]", flush=True)
                (OUT/"repro_report.json").write_text(json.dumps(rows, indent=2))

n = len(rows); npass = sum(r["status"] == "PASS" for r in rows)
mx = max((r["abs_diff"] for r in rows if r["abs_diff"] is not None), default=0)
print(f"\n=== REPRODUCIBILITY: {npass}/{n} cells match committed within {TOL} (max diff {mx:.4f}) ===")
print("VERDICT: REPRODUCIBLE" if npass == n else "VERDICT: DISCREPANCIES - investigate FAIL rows")
