r"""Definitive MIA evaluation for the paper (positive control + split-design sensitivity).
One run produces, for SWaT and SKAB, LiRA + weak AUC (mean+/-std over seeds) for:
  target in {dp(eps=2), np_overfit (non-private, memorising = positive control)}
  split  in {random (uniform)} U {blocked with guard gap in GAPS}
Fixed block size isolates the guard-gap effect. Reading:
  - np_overfit must be HIGH if the attack is capable (positive control);
  - uniform-random collapses every target to chance (the broken design);
  - DP under a gapped blocked split -> its true, controlled leakage level.
Also reports the measured autocorrelation length of each normal stream.
Outputs -> results/mia_privacy_final.json (+ _summary.json), incremental.
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

R_REF=6; SEEDS=5; EPS=2.0; OVERFIT_PASSES=80
BLOCK=1500; GAPS=[0,200,400,600]; NMAX=12000

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

def autocorr_len(Xn, thresh=1/np.e, maxlag=3000):
    sd=Xn.std(0); keep=sd>1e-9
    z=(Xn[:,keep]-Xn[:,keep].mean(0))/sd[keep]; n=len(z)
    for k in range(1,min(maxlag,n-1)):
        ac=np.mean([np.corrcoef(z[:-k,j],z[k:,j])[0,1] for j in range(z.shape[1])])
        if ac<thresh: return k
    return maxlag

def make_split(n, split, gap, rng):
    if split=="contiguous":
        nh=n//2; return np.arange(nh), np.arange(nh,2*nh)
    if split=="random":
        perm=rng.permutation(n); nh=n//2; return perm[:nh], perm[nh:2*nh]
    nb=n//BLOCK; assign=rng.integers(0,2,nb); mem=[]; non=[]
    for b in range(nb):
        s=b*BLOCK+gap; e=(b+1)*BLOCK-gap
        if e<=s: continue
        (mem if assign[b] else non).extend(range(s,e))
    return np.array(mem,int), np.array(non,int)

def attack(Xn,cfg,target,split,gap,seed):
    rng=np.random.default_rng(seed); n=len(Xn)
    mem_idx,non_idx=make_split(n,split,gap,rng)
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
    Ltgt=losses(tgt,Xs); weak=roc_auc_score(label,-Ltgt)
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
    return float(weak), float(roc_auc_score(label,score)), int(m)

CONDS=[("contiguous",None),("random",None)]+[("blocked",g) for g in GAPS]
out=[]; summ=[]
for ds in ["swat","skab"]:
    try: cfg=DS_CFG[ds]; X,y=cached_load(ds,cfg["max_samples"])
    except Exception as e: print(f"SKIP {ds}: {e}",flush=True); continue
    Xn=X[y==0]
    aclen=autocorr_len(Xn)
    if len(Xn)>NMAX: Xn=Xn[:NMAX]
    print(f"[{ds}] normal n={len(Xn)} (autocorr length={aclen}), block={BLOCK}, gaps={GAPS}",flush=True)
    for target in ["dp","np_overfit"]:
        for split,gap in CONDS:
            ws=[]; ls=[]; nm=None
            for seed in range(SEEDS):
                res=attack(Xn,cfg,target,split,gap,seed)
                if res is None: break
                w,l,nm=res; ws.append(w); ls.append(l)
                out.append({"dataset":ds,"target":target,"split":split,"gap":gap,"seed":seed,"weak":round(w,4),"lira":round(l,4),"n_member":nm})
                (ROOT/"results"/"mia_privacy_final.json").write_text(json.dumps(out,indent=2))
                tag=("blocked(gap=%d)"%gap) if split=="blocked" else split
                print(f"  {ds} {target:11s} {tag:16s} seed={seed}: LiRA={l:.3f} weak={w:.3f}",flush=True)
            if ls:
                s={"dataset":ds,"target":target,"split":split,"gap":gap,"autocorr_len":aclen,
                   "lira_mean":round(float(np.mean(ls)),4),"lira_std":round(float(np.std(ls)),4),
                   "weak_mean":round(float(np.mean(ws)),4),"weak_std":round(float(np.std(ws)),4),
                   "seeds":len(ls),"n_member":nm,"block":BLOCK}
                summ.append(s); (ROOT/"results"/"mia_privacy_final_summary.json").write_text(json.dumps(summ,indent=2))
                print(f"  -> {ds} {target} {tag}: LiRA {s['lira_mean']}+/-{s['lira_std']}",flush=True)
print("\nDONE mia_privacy_final.",flush=True)
