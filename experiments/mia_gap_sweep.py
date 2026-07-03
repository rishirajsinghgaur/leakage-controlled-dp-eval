r"""Guard-gap sweep: how does MIA AUC depend on the evaluation split's guard gap?
Fixed block size (1000), vary the guard gap. For each (dataset, target, gap) we run a
blocked member/non-member split and report LiRA AUC (mean+/-std over seeds).
Target = dp(eps=2) or np_overfit (non-private, memorising) as the positive control.
Reading:
  - np_overfit should stay HIGH across gaps if the attack is capable (positive control);
  - if the DP curve swings with the gap, the 'leakage' number is a design artifact.
Outputs -> results/mia_gap_sweep.json (+ _summary.json), incremental.
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

R_REF=6; SEEDS=4; EPS=2.0; OVERFIT_PASSES=80
BLOCK=1000; GAPS=[0,100,200,400,600]; NMAX=9000

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

def blocked_split(n,blk,gap,rng):
    nb=n//blk; assign=rng.integers(0,2,nb); mem=[]; non=[]
    for b in range(nb):
        s=b*blk+gap; e=(b+1)*blk-gap
        if e<=s: continue
        (mem if assign[b] else non).extend(range(s,e))
    return np.array(mem,int), np.array(non,int)

def attack(Xn,cfg,target,gap,seed):
    rng=np.random.default_rng(seed); n=len(Xn)
    mem_idx,non_idx=blocked_split(n,BLOCK,gap,rng)
    m=min(len(mem_idx),len(non_idx))
    if m<40: return None
    mem_idx=mem_idx[:m]; non_idx=non_idx[:m]
    idx=np.concatenate([mem_idx,non_idx]); Xs=Xn[idx]
    label=np.zeros(len(Xs),int); label[:m]=1; Xmem=Xn[mem_idx]
    if target=="dp":
        B=min(cfg["batch_size"],len(Xmem)-1); steps=max(1,len(Xmem)//B)*12
        sig=compute_sigma_for_total_epsilon(EPS,len(Xmem),B,steps,1,cfg["delta"])
        tgt=train_ae(Xmem,cfg,dp_sigma=sig,seed=seed,passes=12)
    else:
        tgt=train_ae(Xmem,cfg,dp_sigma=None,seed=seed,passes=OVERFIT_PASSES)
    Ltgt=losses(tgt,Xs)
    ref_in=[[] for _ in range(len(Xs))]
    for r in range(R_REF):
        ridx=rng.permutation(len(Xs)); half=ridx[:len(Xs)//2]
        refm=train_ae(Xs[half],cfg,dp_sigma=None,seed=1000+10*seed+r,passes=12)
        L=losses(refm,Xs); inset=set(half.tolist())
        for i in range(len(Xs)):
            if i not in inset: ref_in[i].append(L[i])
    score=np.zeros(len(Xs))
    for i in range(len(Xs)):
        o=np.array(ref_in[i]) if ref_in[i] else np.array([Ltgt.mean()])
        mu,sd=o.mean(),o.std()+1e-6; score[i]=(mu-Ltgt[i])/sd
    return float(roc_auc_score(label,score)), int(m)

out=[]; summ=[]
for ds in ["swat","skab"]:
    try: cfg=DS_CFG[ds]; X,y=cached_load(ds,cfg["max_samples"])
    except Exception as e: print(f"SKIP {ds}: {e}",flush=True); continue
    Xn=X[y==0]
    if len(Xn)>NMAX: Xn=Xn[:NMAX]
    print(f"[{ds}] normal n={len(Xn)}, block={BLOCK}, gaps={GAPS}",flush=True)
    for target in ["dp","np_overfit"]:
        for gap in GAPS:
            aucs=[]; nm=None
            for seed in range(SEEDS):
                res=attack(Xn,cfg,target,gap,seed)
                if res is None: break
                a,nm=res; aucs.append(a)
                out.append({"dataset":ds,"target":target,"gap":gap,"seed":seed,"lira":round(a,4),"n_member":nm})
                (ROOT/"results"/"mia_gap_sweep.json").write_text(json.dumps(out,indent=2))
                print(f"  {ds} {target:11s} gap={gap:4d} seed={seed}: LiRA={a:.3f}",flush=True)
            if aucs:
                s={"dataset":ds,"target":target,"gap":gap,"lira_mean":round(float(np.mean(aucs)),4),
                   "lira_std":round(float(np.std(aucs)),4),"seeds":len(aucs),"n_member":nm,"block":BLOCK}
                summ.append(s); (ROOT/"results"/"mia_gap_sweep_summary.json").write_text(json.dumps(summ,indent=2))
                print(f"  -> {ds} {target} gap={gap}: LiRA {s['lira_mean']}+/-{s['lira_std']}",flush=True)
print("\nDONE gap sweep.",flush=True)
