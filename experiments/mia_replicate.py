r"""Independent replication + budget/seed strengthening of the MIA finding.
Re-runs the definitive evaluation with a DIFFERENT seed base (200..) and MORE seeds,
and extends the DP target to all budgets eps in {0.5,1,2,4}. Purpose:
  (1) replicate the positive-control result under independent seeds;
  (2) show the finding holds across privacy budgets, not just eps=2.
Splits: contiguous | uniform-random | blocked (gap 400 > SWaT autocorr 329).
Targets: non-private overfit positive control (eps=inf) and DP at each budget.
Outputs -> results/mia_replicate.json (+ _summary.json), incremental.
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

R_REF=6; SEEDS=6; SEED_BASE=200; OVERFIT_PASSES=80
BLOCK=1500; GAP=400; NMAX=12000
EPS_LIST=[0.5,1.0,2.0,4.0]

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
    if split=="contiguous":
        nh=n//2; return np.arange(nh), np.arange(nh,2*nh)
    if split=="random":
        p=rng.permutation(n); nh=n//2; return p[:nh], p[nh:2*nh]
    nb=n//BLOCK; a=rng.integers(0,2,nb); mem=[]; non=[]
    for b in range(nb):
        s=b*BLOCK+GAP; e=(b+1)*BLOCK-GAP
        if e<=s: continue
        (mem if a[b] else non).extend(range(s,e))
    return np.array(mem,int), np.array(non,int)

def attack(Xn,cfg,kind,eps,split,seed):
    rng=np.random.default_rng(seed); n=len(Xn)
    mi,ni=split_idx(n,split,rng); m=min(len(mi),len(ni))
    if m<40: return None
    mi=mi[:m]; ni=ni[:m]; idx=np.concatenate([mi,ni]); Xs=Xn[idx]
    label=np.zeros(len(Xs),int); label[:m]=1; Xmem=Xn[mi]
    if kind=="dp":
        B=min(cfg["batch_size"],len(Xmem)-1); steps=max(1,len(Xmem)//B)*12
        sig=compute_sigma_for_total_epsilon(eps,len(Xmem),B,steps,1,cfg["delta"])
        tgt=train_ae(Xmem,cfg,dp_sigma=sig,seed=seed,passes=12)
    else:
        tgt=train_ae(Xmem,cfg,dp_sigma=None,seed=seed,passes=OVERFIT_PASSES)
    Ltgt=losses(tgt,Xs)
    ref_in=[[] for _ in range(len(Xs))]
    for r in range(R_REF):
        ridx=rng.permutation(len(Xs)); half=ridx[:len(Xs)//2]
        refm=train_ae(Xs[half],cfg,dp_sigma=None,seed=5000+10*seed+r,passes=12)
        L=losses(refm,Xs); inset=set(half.tolist())
        for i in range(len(Xs)):
            if i not in inset: ref_in[i].append(L[i])
    score=np.zeros(len(Xs))
    for i in range(len(Xs)):
        o=np.array(ref_in[i]) if ref_in[i] else np.array([Ltgt.mean()])
        mu,sd=o.mean(),o.std()+1e-6; score[i]=(mu-Ltgt[i])/sd
    return float(roc_auc_score(label,score)), int(m)

# (kind, eps) configs: positive control once; DP at each budget
CONFIGS=[("np_overfit",None)]+[("dp",e) for e in EPS_LIST]
SPLITS=["contiguous","random","blocked"]
out=[]; summ=[]
for ds in ["swat","skab"]:
    try: cfg=DS_CFG[ds]; X,y=cached_load(ds,cfg["max_samples"])
    except Exception as e: print(f"SKIP {ds}: {e}",flush=True); continue
    Xn=X[y==0]
    if len(Xn)>NMAX: Xn=Xn[:NMAX]
    print(f"[{ds}] n={len(Xn)}, block={BLOCK}, gap={GAP}, seeds {SEED_BASE}..{SEED_BASE+SEEDS-1}",flush=True)
    for kind,eps in CONFIGS:
        for split in SPLITS:
            aucs=[]; nm=None
            for k in range(SEEDS):
                seed=SEED_BASE+k
                res=attack(Xn,cfg,kind,eps,split,seed)
                if res is None: break
                a,nm=res; aucs.append(a)
                out.append({"dataset":ds,"target":kind,"eps":eps,"split":split,"seed":seed,"lira":round(a,4),"n_member":nm})
                (ROOT/"results"/"mia_replicate.json").write_text(json.dumps(out,indent=2))
            if aucs:
                tag=f"{kind}" + (f"@eps{eps}" if eps else "") + f"/{split}"
                s={"dataset":ds,"target":kind,"eps":eps,"split":split,"lira_mean":round(float(np.mean(aucs)),4),
                   "lira_std":round(float(np.std(aucs)),4),"seeds":len(aucs),"n_member":nm}
                summ.append(s); (ROOT/"results"/"mia_replicate_summary.json").write_text(json.dumps(summ,indent=2))
                print(f"  -> {ds} {tag}: LiRA {s['lira_mean']}+/-{s['lira_std']}",flush=True)
print("\nDONE mia_replicate.",flush=True)
