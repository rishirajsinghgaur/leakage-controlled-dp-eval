r"""Positive control + blocked-split robustness for the MIA evaluation.
Answers reviewer Major #1 (positive control) and #2 (autocorrelation / blocked split).

For SWaT (and SKAB if its data is available) we cross:
  target training : dp (eps=2) | np (non-private) | np_overfit (non-private, many passes)
  member/non-member split : random (uniform) | blocked (contiguous blocks assigned
                            randomly to member/non-member, with a guard gap > autocorr length)
Logic: a valid attack apparatus MUST show high AUC for the non-private / overfit target.
If AUC is ~0.5 there too, the near-chance DP result is a property of the evaluation,
not of the DP guarantee. The blocked split tests whether the uniform-random split
deflates AUC via temporal near-duplicates.
Outputs -> results/mia_positive_control.json (+ _summary.json), incremental.
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

R_REF = 8
SEEDS = 3
EPS   = 2.0
OVERFIT_PASSES = 80

def train_ae(X, cfg, dp_sigma=None, seed=0, passes=12):
    torch.manual_seed(seed); np.random.seed(seed)
    n=len(X); B=min(cfg["batch_size"],max(2,n-1))
    m=AnomalyAutoencoder(X.shape[1],cfg["bottleneck"]); opt=torch.optim.Adam(m.parameters(),lr=cfg["lr"])
    ld=DataLoader(TensorDataset(torch.from_numpy(X).float()),batch_size=B,shuffle=True,drop_last=True)
    if dp_sigma is not None:
        from opacus import PrivacyEngine
        m,opt,ld=PrivacyEngine().make_private(module=m,optimizer=opt,data_loader=ld,
                    noise_multiplier=dp_sigma,max_grad_norm=cfg["max_grad_norm"])
    lf=nn.MSELoss(); m.train()
    for _ in range(passes):
        for (b,) in ld:
            opt.zero_grad(); lf(m(b),b).backward(); opt.step()
    return getattr(m,"_module",m)

def losses(model, X):
    model.eval()
    with torch.no_grad():
        r=model(torch.from_numpy(X).float()); return ((r-torch.from_numpy(X).float())**2).mean(1).numpy()

def autocorr_len(Xn, thresh=1/np.e, maxlag=2500):
    sd=Xn.std(0); keep=sd>1e-9                    # drop constant sensors (NaN corr)
    z=(Xn[:,keep]-Xn[:,keep].mean(0))/sd[keep]; n=len(z)
    for k in range(1,min(maxlag,n-1)):
        ac=np.mean([np.corrcoef(z[:-k,j],z[k:,j])[0,1] for j in range(z.shape[1])])
        if ac<thresh: return k, float(ac)
    return maxlag, None

def random_split(n, rng):
    perm=rng.permutation(n); nh=n//2; return perm[:nh], perm[nh:2*nh]

def blocked_split(n, blk, gap, rng):
    nb=n//blk; assign=rng.integers(0,2,nb); mem=[]; non=[]
    for b in range(nb):
        s=b*blk+gap; e=(b+1)*blk-gap
        if e<=s: continue
        (mem if assign[b] else non).extend(range(s,e))
    return np.array(mem,int), np.array(non,int)

def run_attack(Xn, cfg, target_kind, split_kind, seed, blk, gap):
    rng=np.random.default_rng(seed); n=len(Xn)
    mem_idx, non_idx = random_split(n,rng) if split_kind=="random" else blocked_split(n,blk,gap,rng)
    m=min(len(mem_idx),len(non_idx))
    if m<50: return None
    mem_idx=mem_idx[:m]; non_idx=non_idx[:m]
    idx=np.concatenate([mem_idx,non_idx]); Xs=Xn[idx]
    label=np.zeros(len(Xs),int); label[:m]=1; Xmem=Xn[mem_idx]
    if target_kind=="dp":
        B=min(cfg["batch_size"],len(Xmem)-1); steps=max(1,len(Xmem)//B)*12
        sig=compute_sigma_for_total_epsilon(EPS,len(Xmem),B,steps,1,cfg["delta"])
        tgt=train_ae(Xmem,cfg,dp_sigma=sig,seed=seed,passes=12)
    elif target_kind=="np":
        tgt=train_ae(Xmem,cfg,dp_sigma=None,seed=seed,passes=12)
    else:
        tgt=train_ae(Xmem,cfg,dp_sigma=None,seed=seed,passes=OVERFIT_PASSES)
    Ltgt=losses(tgt,Xs)
    weak=roc_auc_score(label,-Ltgt)
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
    lira=roc_auc_score(label,score)
    return float(weak), float(lira), int(m)

out=[]; summ=[]
for ds in ["swat","skab"]:
    try:
        cfg=DS_CFG[ds]; X,y=cached_load(ds,cfg["max_samples"])
    except Exception as e:
        print(f"SKIP {ds}: {type(e).__name__}: {e}",flush=True); continue
    Xn=X[y==0]
    if len(Xn)>9000: Xn=Xn[:9000]
    aclen, acval = autocorr_len(Xn)
    gap=int(np.clip(1.2*aclen, 12, len(Xn)//12))      # guard gap EXCEEDS autocorr length
    blk=int(np.clip(3*gap, 60, len(Xn)//5))
    print(f"[{ds}] normal n={len(Xn)}, autocorr length ~{aclen} (mean ac={acval}), block={blk}, guard gap={gap}",flush=True)
    for target_kind in ["dp","np","np_overfit"]:
        for split_kind in ["random","blocked"]:
            ws=[]; ls=[]; nm=None
            for seed in range(SEEDS):
                res=run_attack(Xn,cfg,target_kind,split_kind,seed,blk,gap)
                if res is None: print(f"  {ds} {target_kind} {split_kind}: too few samples, skip",flush=True); break
                w,l,nm=res; ws.append(w); ls.append(l)
                out.append({"dataset":ds,"target":target_kind,"split":split_kind,"seed":seed,
                            "weak":round(w,4),"lira":round(l,4),"n_member":nm})
                (ROOT/"results"/"mia_positive_control.json").write_text(json.dumps(out,indent=2))
                print(f"  {ds} {target_kind:11s} {split_kind:8s} seed={seed}: weak={w:.3f} LiRA={l:.3f}",flush=True)
            if ws:
                s={"dataset":ds,"target":target_kind,"split":split_kind,
                   "weak_mean":round(float(np.mean(ws)),4),"weak_std":round(float(np.std(ws)),4),
                   "lira_mean":round(float(np.mean(ls)),4),"lira_std":round(float(np.std(ls)),4),
                   "seeds":len(ws),"n_member":nm,"autocorr_len":aclen,"block":blk,"gap":gap}
                summ.append(s); (ROOT/"results"/"mia_positive_control_summary.json").write_text(json.dumps(summ,indent=2))
                print(f"  -> {ds} {target_kind} {split_kind}: weak {s['weak_mean']}+/-{s['weak_std']}  LiRA {s['lira_mean']}+/-{s['lira_std']}",flush=True)
print("\nDONE positive-control + blocked-split.",flush=True)
