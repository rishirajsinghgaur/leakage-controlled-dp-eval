r"""
Stronger membership-inference attack (LiRA-style, per-example calibrated) on the
DP reconstruction detector, vs the weaker shadow-logistic attack. Trains R reference
('shadow') models on random halves; for each probe sample, calibrates its target-model
reconstruction loss against the OUT-distribution of reference losses (Gaussian LR /
z-score). Reports AUC (members vs non-members). Near-0.5 => DP protects even under a
calibrated attack. Real outputs -> results/lira_mia.json.
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
from sklearn.metrics import roc_auc_score
from data.loaders import DATASET_REGISTRY
from models.mlp import AnomalyAutoencoder
from privacy.accountant import compute_sigma_for_total_epsilon
from experiments.run_full_paper_sweep import DS_CFG

R_REF = 8          # reference models
PASSES = 12

def train_ae(X, cfg, dp_sigma=None, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    n=len(X); B=min(cfg["batch_size"],max(2,n-1))
    m=AnomalyAutoencoder(X.shape[1],cfg["bottleneck"]); opt=torch.optim.Adam(m.parameters(),lr=cfg["lr"])
    ld=DataLoader(TensorDataset(torch.from_numpy(X).float()),batch_size=B,shuffle=True,drop_last=True)
    if dp_sigma is not None:
        from opacus import PrivacyEngine
        m,opt,ld=PrivacyEngine().make_private(module=m,optimizer=opt,data_loader=ld,noise_multiplier=dp_sigma,max_grad_norm=cfg["max_grad_norm"])
    lf=nn.MSELoss(); m.train()
    for _ in range(PASSES):
        for (b,) in ld:
            opt.zero_grad(); lf(m(b),b).backward(); opt.step()
    return getattr(m,"_module",m)

def losses(model, X):
    model.eval()
    with torch.no_grad():
        r=model(torch.from_numpy(X).float()); return ((r-torch.from_numpy(X).float())**2).mean(1).numpy()

out=[]
for ds in ["skab","swat"]:
    cfg=DS_CFG[ds]
    from experiments.cached_load import cached_load
    X,y=cached_load(ds,cfg["max_samples"])
    Xn=X[y==0]
    if len(Xn)>4000: Xn=Xn[:4000]
    nh=len(Xn)//2
    for eps in [0.5,2.0]:
        # target model trained on first half (members), DP
        Xmem, Xnon = Xn[:nh], Xn[nh:]
        B=min(cfg["batch_size"],nh-1); steps=max(1,nh//B)*PASSES
        sig=compute_sigma_for_total_epsilon(eps,nh,B,steps,1,cfg["delta"])
        tgt=train_ae(Xmem,cfg,dp_sigma=sig,seed=0)
        # reference models on random halves (non-DP, strong attacker)
        ref_in=[[] for _ in range(len(Xn))]
        rng=np.random.default_rng(0)
        for r in range(R_REF):
            idx=rng.permutation(len(Xn)); half=idx[:nh]; refm=train_ae(Xn[half],cfg,dp_sigma=None,seed=100+r)
            L=losses(refm,Xn)
            inset=set(half.tolist())
            for i in range(len(Xn)):
                if i not in inset: ref_in[i].append(L[i])      # OUT losses for calibration
        Ltgt=losses(tgt,Xn)
        # LiRA-style score: lower loss than OUT-distn => more likely member. z = (mu_out - loss)/sigma_out
        score=np.zeros(len(Xn))
        for i in range(len(Xn)):
            o=np.array(ref_in[i]) if ref_in[i] else np.array([Ltgt.mean()])
            mu,sd=o.mean(),o.std()+1e-6; score[i]=(mu-Ltgt[i])/sd
        label=np.zeros(len(Xn),int); label[:nh]=1
        auc=roc_auc_score(label,score)
        out.append({"dataset":ds,"epsilon":eps,"attack":"LiRA_calibrated","auc":round(float(auc),4),"n_ref":R_REF})
        print(f"{ds} eps={eps}: LiRA-style MIA AUC={auc:.4f}",flush=True)
        (ROOT/"results"/"lira_mia.json").write_text(json.dumps(out,indent=2))
print("\nDone. AUC near 0.5 => DP protects under a calibrated attack.")
