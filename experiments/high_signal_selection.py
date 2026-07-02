r"""
MAKE-OR-BREAK for the floor-effect objection (raised by multiple reviewers).
The rebuttal must show a regime where POINT-WISE F1 is genuinely strong (>0.6) and the
selection null STILL holds — not point-adjusted (inflation-prone), not near the floor.

The 40-epoch normal-only AE reaches point-wise F1 ~0.66 on SWaT WITHOUT DP. Here we run
the full four-arm selection comparison (full/random/tdedup/fps) in that high-signal
regime: non-private, and light-DP (eps=8). If no learned rule beats random/full where the
detector genuinely works, the floor-effect objection is closed.
Real outputs -> results/high_signal_selection.json. NEVER fabricate.
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
    E = E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-12)
    n=len(E); b=min(b,n); sel=[0]; d=1.0-E@E[0]
    for _ in range(1,b):
        i=int(np.argmax(d)); sel.append(i); d=np.minimum(d,1.0-E@E[i])
    return np.array(sorted(sel))

def train_eval(Xt, Xte, yte, cfg, seed, eps=None):
    torch.manual_seed(seed); np.random.seed(seed)
    n=len(Xt); B=min(cfg["batch_size"],max(2,n-1))
    m=AnomalyAutoencoder(Xt.shape[1],cfg["bottleneck"]); opt=torch.optim.Adam(m.parameters(),cfg["lr"])
    ld=DataLoader(TensorDataset(torch.from_numpy(Xt).float()),batch_size=B,shuffle=True,drop_last=True)
    if eps is not None:
        steps=max(1,n//B)*EPOCHS; sig=compute_sigma_for_total_epsilon(eps,n,B,steps,1,cfg["delta"])
        from opacus import PrivacyEngine
        m,opt,ld=PrivacyEngine().make_private(module=m,optimizer=opt,data_loader=ld,noise_multiplier=sig,max_grad_norm=cfg["max_grad_norm"])
    m.train()
    for _ in range(EPOCHS):
        for (b,) in ld: opt.zero_grad(); nn.MSELoss()(m(b),b).backward(); opt.step()
    mm=getattr(m,"_module",m); sc=mm.anomaly_score(Xte)
    thr=np.percentile(sc,cfg["eval_percentile"])
    return f1_score(yte,(sc>=thr).astype(int),zero_division=0)   # POINT-WISE only

out=[]
for ds in ["swat","skab"]:
    cfg=DS_CFG[ds]
    X,y=cached_load(ds,cfg["max_samples"])
    idx=np.arange(len(X)); tr,te=train_test_split(idx,test_size=0.2,random_state=0,stratify=y)
    tr=np.sort(tr); Xtr,ytr,Xte,yte=X[tr],y[tr],X[te],y[te]
    enc=SiameseEncoder(Xtr.shape[1],cfg["siamese_emb"]); enc=train_siamese(enc,Xtr,ytr,epochs=6,window=cfg["siamese_window"])
    Xn=Xtr[ytr==0]
    if len(Xn)>6000: Xn=Xn[:6000]
    E=enc.encode(Xn); kept,_=seq_temporal_dedup(Xn,enc,keep_quantile=0.5); Xd=Xn[kept]; bud=len(Xd); Xf=Xn[fps(E,bud)]
    print(f"{ds}: normal={len(Xn)} budget={bud}",flush=True)
    for cond,eps in [("nonprivate",None),("eps8",8.0)]:
        for seed in range(5):
            rng=np.random.default_rng(1000+seed); Xr=Xn[np.sort(rng.choice(len(Xn),bud,replace=False))]
            for mode,Xt in [("full",Xn),("random",Xr),("tdedup",Xd),("fps",Xf)]:
                f1=train_eval(Xt,Xte,yte,cfg,seed,eps=eps)
                out.append({"dataset":ds,"condition":cond,"mode":mode,"seed":seed,"f1":round(float(f1),4),"n":len(Xt)})
                (ROOT/"results"/"high_signal_selection.json").write_text(json.dumps(out,indent=2))
            print(f"  {ds} {cond} seed={seed} done",flush=True)
print("\n=== HIGH-SIGNAL SELECTION SUMMARY (point-wise F1 mean) ===")
for ds in ["swat","skab"]:
    for cond in ["nonprivate","eps8"]:
        r={m:np.mean([x["f1"] for x in out if x["dataset"]==ds and x["condition"]==cond and x["mode"]==m]) for m in ["full","random","tdedup","fps"]}
        print(f"{ds} {cond}: "+" ".join(f"{m}={r[m]:.3f}" for m in r)+f" | spread={max(r.values())-min(r.values()):.3f}")
