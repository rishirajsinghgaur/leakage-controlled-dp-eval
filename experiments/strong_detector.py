r"""
DECISIVE experiment (reviewer floor-effect critique): does a STRONGER detector,
operating well above the random-score floor, change the selection conclusion?

Strong detector = sliding-window deep autoencoder (stacks w consecutive timesteps so
the detector sees temporal dynamics; this is the standard big F1 lift for time-series
anomaly detection over a per-sample AE). Window w, deeper/wider AE.

Stage A (feasibility, NON-private): confirm the strong detector reaches F1 well above
the floor (SKAB ~0.149 / SWaT ~0.130) and above the per-sample AE (~0.21 / ~0.33).
Stage B (DP + selection): if Stage A clears the floor, test full vs random vs temporal
vs diversity selection under DP with the strong detector.

Real outputs -> results/strong_detector.json. NEVER fabricate.
"""
import sys, json, logging, argparse
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.ERROR)
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score
from experiments.cached_load import cached_load
from experiments.run_full_paper_sweep import DS_CFG
from privacy.accountant import compute_sigma_for_total_epsilon


def windowize(X, w):
    if w <= 1: return X
    n = len(X) - w + 1
    return np.stack([X[i:i+w].reshape(-1) for i in range(n)], axis=0)


class DeepAE(nn.Module):
    def __init__(self, d, h=256, z=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d,h), nn.ReLU(), nn.Linear(h,h//2), nn.ReLU(), nn.Linear(h//2,z))
        self.dec = nn.Sequential(nn.Linear(z,h//2), nn.ReLU(), nn.Linear(h//2,h), nn.ReLU(), nn.Linear(h,d))
    def forward(self,x): return self.dec(self.enc(x))


def train_eval(Xtr_n, Xte, yte, w, epochs, dp_sigma=None, seed=0, lr=1e-3, bs=128):
    torch.manual_seed(seed); np.random.seed(seed)
    Xw = windowize(Xtr_n, w); d = Xw.shape[1]
    m = DeepAE(d); opt = torch.optim.Adam(m.parameters(), lr=lr)
    B = min(bs, max(2, len(Xw)-1))
    ld = DataLoader(TensorDataset(torch.from_numpy(Xw).float()), batch_size=B, shuffle=True, drop_last=True)
    if dp_sigma is not None:
        from opacus import PrivacyEngine
        m, opt, ld = PrivacyEngine().make_private(module=m, optimizer=opt, data_loader=ld,
                                                  noise_multiplier=dp_sigma, max_grad_norm=1.0)
    lf = nn.MSELoss(); m.train()
    for _ in range(epochs):
        for (b,) in ld:
            opt.zero_grad(); lf(m(b), b).backward(); opt.step()
    mm = getattr(m, "_module", m); mm.eval()
    Xtew = windowize(Xte, w); ytew = yte[w-1:] if w > 1 else yte
    with torch.no_grad():
        r = mm(torch.from_numpy(Xtew).float()); sc = ((r-torch.from_numpy(Xtew).float())**2).mean(1).numpy()
    return sc, ytew


def f1_at(sc, y, pct):
    thr = np.percentile(sc, pct); yp = (sc >= thr).astype(int)
    return f1_score(y, yp, zero_division=0), average_precision_score(y, sc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="A")          # A=feasibility non-DP, B=DP+selection
    ap.add_argument("--datasets", nargs="+", default=["swat","skab"])
    ap.add_argument("--w", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    out = []
    for ds in a.datasets:
        cfg = DS_CFG[ds]; pct = cfg["eval_percentile"]
        X, y = cached_load(ds, cfg["max_samples"])
        idx = np.arange(len(X)); tr, te = train_test_split(idx, test_size=0.2, random_state=0, stratify=y); tr = np.sort(tr)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        Xn = Xtr[ytr == 0]
        if a.stage == "A":
            for seed in (0,1,2):
                sc, yy = train_eval(Xn, Xte, yte, a.w, a.epochs, dp_sigma=None, seed=seed)
                f1, ap_ = f1_at(sc, yy, pct)
                out.append({"dataset":ds,"stage":"A_nonDP","w":a.w,"seed":seed,"f1":round(f1,4),"auprc":round(ap_,4)})
                print(f"{ds} non-DP w={a.w} s={seed}: F1={f1:.4f} AUPRC={ap_:.4f}", flush=True)
        (ROOT/"results"/"strong_detector.json").write_text(json.dumps(out,indent=2))
    print("\nSUMMARY:", {ds: round(np.mean([o['f1'] for o in out if o['dataset']==ds]),3) for ds in a.datasets})


if __name__ == "__main__":
    main()
