r"""
Validation (reviewer concern): is the near-chance randomized-split MIA AUC an artifact of using
only N=8 reference models (LiRA standard is 64-128)? We re-run SWaT eps=2 with N=64,
both the contiguous and randomized split, to confirm the artifact and
the chance result are stable at higher N. Output -> results/lira_n64_check.json.
"""
import sys, json, logging
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.ERROR)
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from models.mlp import AnomalyAutoencoder
from privacy.accountant import compute_sigma_for_total_epsilon
from experiments.run_full_paper_sweep import DS_CFG
from experiments.cached_load import cached_load

R_REF = 64; PASSES = 12; SEEDS = 3; ds = "swat"; eps = 2.0
cfg = DS_CFG[ds]

def train(X, sig=None, seed=0):
    torch.manual_seed(seed); np.random.seed(seed); n=len(X); B=min(cfg["batch_size"],max(2,n-1))
    m=AnomalyAutoencoder(X.shape[1],cfg["bottleneck"]); opt=torch.optim.Adam(m.parameters(),cfg["lr"])
    ld=DataLoader(TensorDataset(torch.from_numpy(X).float()),batch_size=B,shuffle=True,drop_last=True)
    if sig is not None:
        from opacus import PrivacyEngine
        m,opt,ld=PrivacyEngine().make_private(module=m,optimizer=opt,data_loader=ld,noise_multiplier=sig,max_grad_norm=cfg["max_grad_norm"])
    for _ in range(PASSES):
        for (b,) in ld: opt.zero_grad(); nn.MSELoss()(m(b),b).backward(); opt.step()
    return getattr(m,"_module",m)

def loss(m,X):
    m.eval()
    with torch.no_grad(): r=m(torch.from_numpy(X).float()); return ((r-torch.from_numpy(X).float())**2).mean(1).numpy()

X,y=cached_load(ds,cfg["max_samples"]); Xn=X[y==0]
if len(Xn)>4000: Xn=Xn[:4000]
nh=len(Xn)//2; B=min(cfg["batch_size"],nh-1); steps=max(1,nh//B)*PASSES
sig=compute_sigma_for_total_epsilon(eps,nh,B,steps,1,cfg["delta"])
out=[]
for split in ["contiguous","randomized"]:
    lr=[]
    for seed in range(SEEDS):
        rng=np.random.default_rng(seed)
        Xs = Xn[rng.permutation(len(Xn))] if split=="randomized" else Xn
        tgt=train(Xs[:nh],sig,seed); Lt=loss(tgt,Xs); lab=np.zeros(len(Xs),int); lab[:nh]=1
        ref_in=[[] for _ in range(len(Xs))]
        for r in range(R_REF):
            idx=rng.permutation(len(Xs)); half=idx[:nh]; rm=train(Xs[half],None,100+10*seed+r)
            L=loss(rm,Xs); ins=set(half.tolist())
            for i in range(len(Xs)):
                if i not in ins: ref_in[i].append(L[i])
        sc=np.array([(np.mean(ref_in[i])-Lt[i])/(np.std(ref_in[i])+1e-6) if ref_in[i] else 0.0 for i in range(len(Xs))])
        a=roc_auc_score(lab,sc); lr.append(a)
        print(f"{split} N=64 seed={seed}: LiRA AUC={a:.3f}",flush=True)
    out.append({"split":split,"n_ref":R_REF,"lira_mean":round(float(np.mean(lr)),4),"lira_std":round(float(np.std(lr)),4)})
    (ROOT/"results"/"lira_n64_check.json").write_text(json.dumps(out,indent=2))
print("DONE:", out)
