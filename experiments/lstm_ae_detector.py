r"""
Published sequence detector (LSTM encoder-decoder, Malhotra et al. 2016 "EncDec-AD"), run
under the same protocol to address the reviewer point that the only sequence model tried was
a weak in-house sliding-window AE. This is a standard, citable architecture.

Stage A (this script, NON-PRIVATE feasibility): does a real LSTM-AE clear the point-wise F1
ceiling (~0.43 non-private) that the per-sample AE could not? Train on normal-only windows,
score each window by reconstruction MSE (assigned to the window's last timestep), evaluate
point-wise F1 at the dataset's eval percentile. If it clears >0.6, Stage B (DP + selection)
is warranted; if not, that a PUBLISHED sequence detector also caps low is far stronger
evidence than the in-house window-AE.

SKAB, SWaT. Output -> results/lstm_ae_detector.json.
"""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.stdout.reconfigure(encoding="utf-8")
import logging; logging.basicConfig(level=logging.ERROR)
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score
from experiments.run_full_paper_sweep import DS_CFG

WIN = 20; HIDDEN = 48; EPOCHS = 25; SEEDS = [0, 1, 2]
_DATASETS = sys.argv[1:] if len(sys.argv) > 1 else ["skab", "swat"]
_OUTP = ROOT / "results" / "lstm_ae_detector.json"
_done = json.load(open(_OUTP)) if _OUTP.exists() else []


def windows(X, w):
    n = len(X) - w + 1
    if n <= 0: return np.empty((0, w, X.shape[1]), np.float32), np.empty(0, int)
    idx = np.arange(n)
    W = np.stack([X[i:i+w] for i in idx], 0).astype(np.float32)
    end = idx + w - 1            # each window scored at its last timestep
    return W, end


class LSTMAE(nn.Module):
    def __init__(self, d, h=48):
        super().__init__()
        self.enc = nn.LSTM(d, h, batch_first=True)
        self.dec = nn.LSTM(h, h, batch_first=True)
        self.out = nn.Linear(h, d)
    def forward(self, x):
        _, (hn, _) = self.enc(x)                       # hn: (1,B,h)
        z = hn[-1].unsqueeze(1).repeat(1, x.size(1), 1)  # (B,w,h) repeat latent
        dec, _ = self.dec(z)
        return self.out(dec)


def run(ds, cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    z = np.load(ROOT/"results"/f"cache_{ds}.npz") if (ROOT/"results"/f"cache_{ds}.npz").exists() else None
    if z is not None:
        X, y = z["X"].astype("float32"), z["y"].astype("int64")
    else:
        from data.loaders import DATASET_REGISTRY
        X, y, _ = DATASET_REGISTRY[ds](max_samples=cfg["max_samples"], random_state=42)
    idx = np.arange(len(X)); tr, te = train_test_split(idx, test_size=0.2, random_state=0, stratify=y); tr = np.sort(tr); te = np.sort(te)
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    Xn = Xtr[ytr == 0]
    if len(Xn) > 6000: Xn = Xn[:6000]
    Wn, _ = windows(Xn, WIN)
    d = X.shape[1]
    m = LSTMAE(d, HIDDEN); opt = torch.optim.Adam(m.parameters(), lr=1e-3); lf = nn.MSELoss()
    ld = DataLoader(TensorDataset(torch.from_numpy(Wn)), batch_size=64, shuffle=True, drop_last=True)
    m.train()
    for ep in range(EPOCHS):
        for (b,) in ld:
            opt.zero_grad(); r = m(b); lf(r, b).backward(); opt.step()
    # score test windows
    m.eval()
    Wte, end = windows(Xte, WIN)
    with torch.no_grad():
        scores = np.zeros(len(Wte))
        for i in range(0, len(Wte), 256):
            b = torch.from_numpy(Wte[i:i+256])
            r = m(b); scores[i:i+256] = ((r-b)**2).mean(dim=(1,2)).numpy()
    yend = yte[end]
    thr = np.percentile(scores, cfg["eval_percentile"])
    pred = (scores >= thr).astype(int)
    f1 = f1_score(yend, pred, zero_division=0)
    ap = average_precision_score(yend, scores) if yend.sum() > 0 else 0.0
    return float(f1), float(ap), int(yend.sum()), len(Wte)


def main():
    out = list(_done)
    for ds in _DATASETS:
        cfg = DS_CFG[ds]
        for seed in SEEDS:
            f1, ap, na, nw = run(ds, cfg, seed)
            out.append({"dataset": ds, "stage": "A_nonDP", "win": WIN, "hidden": HIDDEN,
                        "epochs": EPOCHS, "seed": seed, "f1": round(f1, 4), "auprc": round(ap, 4),
                        "n_anom": na, "n_win": nw})
            print(f"  {ds} LSTM-AE s={seed} F1={f1:.4f} AUPRC={ap:.4f} (win={WIN})", flush=True)
            _OUTP.write_text(json.dumps(out, indent=2))
    print("\n=== LSTM-AE (non-private) point-wise F1 vs the ~0.43 ceiling ===")
    for ds in sorted(set(r["dataset"] for r in out)):
        v = [r["f1"] for r in out if r["dataset"] == ds]
        print(f"  {ds}: {np.mean(v):.3f} +/- {np.std(v):.3f}")


if __name__ == "__main__":
    main()
