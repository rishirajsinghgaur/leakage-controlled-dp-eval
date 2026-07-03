r"""QUICK author self-check of the MIA reframe (~5-8 min, SWaT only, 2 seeds).
Reproduces the three key facts with your own eyes:
  1. uniform-random split is BLIND  (both DP and a memorised model score ~0.50)
  2. blocked split PASSES the positive control (memorised model ~0.99)
  3. under the blocked split DP sits well BELOW that ceiling
Run:  python experiments/verify_mia_quick.py
This is a smaller/faster version of experiments/mia_privacy_final.py; the paper's
numbers come from that full script (5 seeds, both datasets).
"""
import sys, logging
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

SEEDS=2; R_REF=4; BLOCK=1500; GAP=400; EPS=2.0

def train_ae(X,cfg,dp_sigma=None,seed=0,passes=12):
    torch.manual_seed(seed); np.random.seed(seed)
    n=len(X); B=min(cfg["batch_size"],max(2,n-1))
    m=AnomalyAutoencoder(X.shape[1],cfg["bottleneck"]); opt=torch.optim.Adam(m.parameters(),lr=cfg["lr"])
    ld=DataLoader(TensorDataset(torch.from_numpy(X).float()),batch_size=B,shuffle=True,drop_last=True)
    if dp_sigma is not None:
        from opacus import PrivacyEngine
        m,opt,ld=PrivacyEngine().make_private(module=m,optimizer=opt,data_loader=ld,noise_multiplier=dp_sigma,max_grad_norm=cfg["max_grad_norm"])
    lf=nn.MSELoss(); m.train()
    for _ in range(passes):
        for (b,) in ld:
            opt.zero_grad(); lf(m(b),b).backward(); opt.step()
    return getattr(m,"_module",m)

def losses(model,X):
    model.eval()
    with torch.no_grad():
        r=model(torch.from_numpy(X).float()); return ((r-torch.from_numpy(X).float())**2).mean(1).numpy()

def split_idx(n,split,rng):
    if split=="random":
        p=rng.permutation(n); nh=n//2; return p[:nh], p[nh:2*nh]
    nb=n//BLOCK; a=rng.integers(0,2,nb); mem=[]; non=[]
    for b in range(nb):
        s=b*BLOCK+GAP; e=(b+1)*BLOCK-GAP
        if e<=s: continue
        (mem if a[b] else non).extend(range(s,e))
    return np.array(mem,int), np.array(non,int)

def auc(Xn,cfg,kind,split,seed):
    rng=np.random.default_rng(seed); n=len(Xn)
    mi,ni=split_idx(n,split,rng); m=min(len(mi),len(ni)); mi,ni=mi[:m],ni[:m]
    idx=np.concatenate([mi,ni]); Xs=Xn[idx]; label=np.zeros(len(Xs),int); label[:m]=1
    if kind=="dp":
        B=min(cfg["batch_size"],m-1); steps=max(1,m//B)*12
        sig=compute_sigma_for_total_epsilon(EPS,m,B,steps,1,cfg["delta"])
        tgt=train_ae(Xn[mi],cfg,dp_sigma=sig,seed=seed,passes=12)
    else:
        tgt=train_ae(Xn[mi],cfg,dp_sigma=None,seed=seed,passes=80)
    Lt=losses(tgt,Xs); ref=[[] for _ in range(len(Xs))]
    for r in range(R_REF):
        h=rng.permutation(len(Xs))[:len(Xs)//2]; rm=train_ae(Xs[h],cfg,dp_sigma=None,seed=900+10*seed+r,passes=12)
        L=losses(rm,Xs); inset=set(h.tolist())
        for i in range(len(Xs)):
            if i not in inset: ref[i].append(L[i])
    sc=np.array([( (np.mean(ref[i]) if ref[i] else Lt.mean()) - Lt[i])/((np.std(ref[i]) if ref[i] else 1)+1e-6) for i in range(len(Xs))])
    return roc_auc_score(label,sc)

cfg=DS_CFG["swat"]; X,y=cached_load("swat",cfg["max_samples"]); Xn=X[y==0][:12000]
print(f"SWaT normal n={len(Xn)} | running (~5-8 min)...\n")
res={}
for kind in ["np_overfit","dp"]:
    for split in ["random","blocked"]:
        vals=[auc(Xn,cfg,kind,split,s) for s in range(SEEDS)]
        res[(kind,split)]=np.mean(vals)
        print(f"  {kind:11s} {split:8s}: LiRA AUC = {np.mean(vals):.3f}")
print("\n--- interpretation ---")
print(f"1. uniform-random BLIND?  memorised model random = {res[('np_overfit','random')]:.2f}  "
      f"(should be ~0.50 -> split cannot detect even a memorised model): "
      f"{'PASS' if res[('np_overfit','random')]<0.6 else 'CHECK'}")
print(f"2. positive control PASSES on blocked?  memorised blocked = {res[('np_overfit','blocked')]:.2f}  "
      f"(should be high, ~0.9+): {'PASS' if res[('np_overfit','blocked')]>0.85 else 'CHECK'}")
print(f"3. DP below the ceiling on blocked?  DP blocked = {res[('dp','blocked')]:.2f}  vs ceiling "
      f"{res[('np_overfit','blocked')]:.2f}: {'PASS' if res[('dp','blocked')] < res[('np_overfit','blocked')]-0.1 else 'CHECK'}")
print("\n(2 seeds only -> numbers are noisier than the paper's 5-6 seed runs; the PATTERN is the check.)")
