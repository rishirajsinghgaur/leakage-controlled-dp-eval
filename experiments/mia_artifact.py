r"""
The MEMBERSHIP-INFERENCE ARTIFACT (second case study).
A protocol-violating MIA labels the first temporal half of the normal stream as
'members' and the second half as 'non-members'. On a drifting process this confounds
membership with temporal distribution shift, producing apparent leakage. We run the
contiguous split with seeds, for both a weak loss-threshold attack and a LiRA
calibrated attack, to quantify the spurious AUC. The randomized (shuffled, seeded) split is
in lira_mia.json and gives chance. Real outputs -> results/mia_artifact.json.
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

R_REF = 8; PASSES = 12; SEEDS = 5

def train_ae(X, cfg, sig=None, seed=0):
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

out=[]; summ=[]
for ds in ["skab","swat"]:
    cfg=DS_CFG[ds]; X,y=cached_load(ds,cfg["max_samples"]); Xn=X[y==0]
    if len(Xn)>4000: Xn=Xn[:4000]
    nh=len(Xn)//2; lab=np.zeros(len(Xn),int); lab[:nh]=1   # CONTIGUOUS: first half = members (the artifact)
    for eps in [0.5,2.0]:
        B=min(cfg["batch_size"],nh-1); steps=max(1,nh//B)*PASSES; sig=compute_sigma_for_total_epsilon(eps,nh,B,steps,1,cfg["delta"])
        wk=[]; lr=[]
        for seed in range(SEEDS):
            tgt=train_ae(Xn[:nh],cfg,sig=sig,seed=seed); Lt=loss(tgt,Xn)
            wk.append(roc_auc_score(lab,-Lt))
            ref_in=[[] for _ in range(len(Xn))]; rng=np.random.default_rng(seed)
            for r in range(R_REF):
                idx=rng.permutation(len(Xn)); half=idx[:nh]; rm=train_ae(Xn[half],cfg,sig=None,seed=100+10*seed+r)
                L=loss(rm,Xn); ins=set(half.tolist())
                for i in range(len(Xn)):
                    if i not in ins: ref_in[i].append(L[i])
            sc=np.array([(np.mean(ref_in[i])-Lt[i])/(np.std(ref_in[i])+1e-6) if ref_in[i] else 0.0 for i in range(len(Xn))])
            lr.append(roc_auc_score(lab,sc))
            out.append({"dataset":ds,"epsilon":eps,"seed":seed,"split":"contiguous_naive","weak_auc":round(float(wk[-1]),4),"lira_auc":round(float(lr[-1]),4)})
            (ROOT/"results"/"mia_artifact.json").write_text(json.dumps(out,indent=2))
            print(f"{ds} eps={eps} seed={seed}: weak={wk[-1]:.3f} LiRA={lr[-1]:.3f}",flush=True)
        summ.append({"dataset":ds,"epsilon":eps,"weak_mean":round(float(np.mean(wk)),4),"weak_std":round(float(np.std(wk)),4),"lira_mean":round(float(np.mean(lr)),4),"lira_std":round(float(np.std(lr)),4)})
        print(f"  -> {ds} eps={eps}: weak {summ[-1]['weak_mean']}+/-{summ[-1]['weak_std']} LiRA {summ[-1]['lira_mean']}+/-{summ[-1]['lira_std']}",flush=True)
(ROOT/"results"/"mia_artifact_summary.json").write_text(json.dumps(summ,indent=2))
print("\nDONE MIA artifact (contiguous split).")
