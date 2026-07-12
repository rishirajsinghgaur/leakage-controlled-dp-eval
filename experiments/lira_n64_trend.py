r"""
N=64 reference-model check of the epsilon-TREND (not just the split effect).

The existing N=64 check (lira_n64_check.py) confirms the split-level behaviour at N=64
(contiguous inflates, uniform-random returns chance) but does not cover the budget trend.
This runs the blocked-split (gap 400) DP attack at the budget extremes eps in {0.5, 4.0}
with N=64 reference models (vs 6 in the main replication), on SWaT, to check that the
membership signal is still larger at looser budget when the reference count is raised 10x.
Bounded scope (2 budgets, 3 seeds) for tractability. Output -> results/lira_n64_trend.json.
"""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.stdout.reconfigure(encoding="utf-8")
import experiments.mia_replicate as M   # reuse train_ae/losses/split_idx/attack machinery
from experiments.run_full_paper_sweep import DS_CFG
from experiments.cached_load import cached_load

M.R_REF = 64                      # override reference-model count
EPS = [0.5, 4.0]; SEEDS = [300, 301, 302]; DS = "swat"
_OUTP = ROOT / "results" / "lira_n64_trend.json"
_done = json.load(open(_OUTP)) if _OUTP.exists() else []
_seen = {(r["eps"], r["seed"]) for r in _done}


def main():
    cfg = DS_CFG[DS]
    X, y = cached_load(DS, cfg["max_samples"])
    Xn = X[y == 0][:M.NMAX]
    out = list(_done)
    print(f"{DS}: normal={len(Xn)}, N_ref={M.R_REF}, block={M.BLOCK}, gap={M.GAP}", flush=True)
    for eps in EPS:
        for seed in SEEDS:
            if (eps, seed) in _seen:
                continue
            res = M.attack(Xn, cfg, "dp", eps, "blocked", seed)
            if res is None:
                continue
            auc, nm = res
            out.append({"dataset": DS, "eps": eps, "seed": seed, "n_ref": 64,
                        "lira": round(auc, 4), "n_member": nm})
            print(f"  eps={eps} seed={seed} N=64 LiRA={auc:.4f} (n_member={nm})", flush=True)
            _OUTP.write_text(json.dumps(out, indent=2))
    print("\n=== N=64 epsilon-trend (SWaT, blocked gap400) ===")
    for eps in EPS:
        v = [r["lira"] for r in out if r["eps"] == eps]
        if v:
            print(f"  eps={eps}: LiRA={np.mean(v):.3f} +/- {np.std(v):.3f}  (n={len(v)})")


if __name__ == "__main__":
    main()
