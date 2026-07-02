r"""
DECISIVE experiment (defeats the floor-effect objection): with a PROPERLY-TRAINED
detector (40 epochs, normal-only) that reaches a high-signal regime (SWaT point-wise
F1~0.66 non-DP), does the choice of training subset matter under DP?

Compares full / random / temporal-dedup / diversity-coreset at matched size, under DP,
reporting BOTH point-wise and point-adjusted F1 (the protocol the strong-detector
literature uses). If selection still does not beat random/full here, the null is robust
at high signal (not a floor artifact). If it does, a real effect was hidden by
under-training.

Real outputs -> results/strong_selection.json. NEVER fabricate.
"""
import sys, json, logging
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.ERROR)
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from experiments.cached_load import cached_load
from experiments.run_full_paper_sweep import DS_CFG
from models.mlp import AnomalyAutoencoder
from models.siamese import SiameseEncoder, train_siamese
from privacy.accountant import compute_sigma_for_total_epsilon
from experiments.principled_method import seq_temporal_dedup

EPOCHS = 40

def fps(E, b):
    n=len(E); b=min(b,n); sel=[0]; d=1.0-E@E[0]
    for _ in range(1,b):
        i=int(np.argmax(d)); sel.append(i); d=np.minimum(d,1.0-E@E[i])
    return np.array(sorted(sel))

def point_adjust(y, yp):
    yp=yp.copy(); i=0; n=len(y)
    while i<n:
        if y[i]==1:
            j=i
            while j<n and y[j]==1: j+=1
            if yp[i:j].any(): yp[i:j]=1
            i=j
        else: i+=1
    return yp

def dp_train_eval(Xt, Xte, yte, cfg, eps, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    n=len(Xt); B=min(cfg["batch_size"],max(2,n-1)); steps=max(1,n//B)*EPOCHS
    sig=compute_sigma_for_total_epsilon(eps,n,B,steps,1,cfg["delta"])
    from opacus import PrivacyEngine
    m=AnomalyAutoencoder(Xt.shape[1],cfg["bottleneck"]); opt=torch.optim.Adam(m.parameters(),cfg["lr"])
    ld=DataLoader(TensorDataset(torch.from_numpy(Xt).float()),batch_size=B,shuffle=True,drop_last=True)
    m,opt,ld=PrivacyEngine().make_private(module=m,optimizer=opt,data_loader=ld,noise_multiplier=sig,max_grad_norm=cfg["max_grad_norm"])
    m.train()
    for _ in range(EPOCHS):
        for (b,) in ld: opt.zero_grad(); nn.MSELoss()(m(b),b).backward(); opt.step()
    mm=getattr(m,"_module",m); sc=mm.anomaly_score(Xte)
    thr=np.percentile(sc,cfg["eval_percentile"]); yp=(sc>=thr).astype(int)
    return f1_score(yte,yp,zero_division=0), f1_score(yte,point_adjust(yte,yp),zero_division=0)

out=[]
for ds in ["swat","skab"]:
    cfg=DS_CFG[ds]
    X,y=cached_load(ds,cfg["max_samples"])
    idx=np.arange(len(X)); tr,te=train_test_split(idx,test_size=0.2,random_state=0,stratify=y)
    tr=np.sort(tr); te=np.sort(te)                    # temporal order for point-adjustment
    Xtr,ytr,Xte,yte=X[tr],y[tr],X[te],y[te]
    enc=SiameseEncoder(Xtr.shape[1],cfg["siamese_emb"]); enc=train_siamese(enc,Xtr,ytr,epochs=6,window=cfg["siamese_window"])
    Xn=Xtr[ytr==0]
    if len(Xn)>6000: Xn=Xn[:6000]
    E=enc.encode(Xn); kept,_=seq_temporal_dedup(Xn,enc); Xd=Xn[kept]; bud=len(Xd); Xf=Xn[fps(E,bud)]
    print(f"{ds}: normal={len(Xn)} budget={bud}",flush=True)
    for seed in (0,1,2):
        rng=np.random.default_rng(1000+seed); Xr=Xn[np.sort(rng.choice(len(Xn),bud,replace=False))]
        for eps in (0.5,2.0):
            for mode,Xt in [("full",Xn),("random",Xr),("tdedup",Xd),("fps",Xf)]:
                pw,pa=dp_train_eval(Xt,Xte,yte,cfg,eps,seed)
                out.append({"dataset":ds,"mode":mode,"epsilon":eps,"seed":seed,"f1_pw":round(pw,4),"f1_pa":round(pa,4),"n":len(Xt)})
                print(f"  {ds} {mode} eps={eps} s={seed}: F1pw={pw:.4f} F1pa={pa:.4f}",flush=True)
                (ROOT/"results"/"strong_selection.json").write_text(json.dumps(out,indent=2))
print("\n=== STRONG-DETECTOR SELECTION SUMMARY (point-wise F1 mean) ===")
for ds in ["swat","skab"]:
    for eps in (0.5,2.0):
        r={m:np.mean([x["f1_pw"] for x in out if x["dataset"]==ds and x["mode"]==m and x["epsilon"]==eps]) for m in ["full","random","tdedup","fps"]}
        sp=max(r.values())-min(r.values())
        print(f"{ds} eps={eps}: "+" ".join(f"{m}={r[m]:.3f}" for m in r)+f" | spread={sp:.3f}")
