r"""Independent memorization check for the non-private overfit positive control: if it has
memorised its training set, its reconstruction loss on TRAIN records is far below its loss on
held-out TEST records. Confirms the positive control is a valid 'memorised' target,
independent of the LiRA attack. SWaT, 5 seeds. Output -> results/memorization_check.json."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.stdout.reconfigure(encoding="utf-8")
import logging; logging.basicConfig(level=logging.ERROR)
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.mlp import AnomalyAutoencoder
from experiments.run_full_paper_sweep import DS_CFG

def run(ds="swat", seed=0, epochs=80):
    torch.manual_seed(seed); np.random.seed(seed)
    z = np.load(ROOT/"results"/f"cache_{ds}.npz")
    X, y = z["X"].astype("float32"), z["y"].astype("int64")
    Xn = X[y == 0]
    rng = np.random.default_rng(seed); idx = rng.permutation(len(Xn))
    ntr = min(3000, len(Xn)//2)
    tr, te = Xn[idx[:ntr]], Xn[idx[ntr:2*ntr]]           # disjoint normal train / held-out
    cfg = DS_CFG[ds]
    m = AnomalyAutoencoder(X.shape[1], cfg["bottleneck"]); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    ld = DataLoader(TensorDataset(torch.from_numpy(tr)), batch_size=64, shuffle=True, drop_last=True)
    lf = nn.MSELoss(); m.train()
    for _ in range(epochs):
        for (b,) in ld:
            opt.zero_grad(); lf(m(b), b).backward(); opt.step()
    m.eval()
    ltr = float(np.mean(m.anomaly_score(tr))); lte = float(np.mean(m.anomaly_score(te)))
    return ltr, lte

out = []
for s in range(5):
    ltr, lte = run("swat", s)
    out.append({"seed": s, "train_loss": round(ltr, 5), "test_loss": round(lte, 5),
                "ratio_test_over_train": round(lte/ltr, 2)})
    print(f"  seed={s} train={ltr:.4f} test={lte:.4f} ratio={lte/ltr:.2f}x", flush=True)
(ROOT/"results"/"memorization_check.json").write_text(json.dumps(out, indent=2))
tr = np.mean([r["train_loss"] for r in out]); te = np.mean([r["test_loss"] for r in out])
print(f"\nSWaT overfit positive control: train MSE {tr:.4f} vs held-out {te:.4f} = {te/tr:.1f}x higher on held-out")
