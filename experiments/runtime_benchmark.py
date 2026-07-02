r"""
Runtime / scalability benchmark (reviewer ask: 'no runtime analysis').
Measures, per client, the wall-clock of each stage and the communication cost:
  S0 (SIL gate fit+flag), S1 (encode + HNSW + union-find dedup), S2 (one DP-SGD round),
  peak memory (S1), and per-round communication bytes (= 2 * n_params * 4).
Also a small scaling probe: S1 dedup time vs n (per client).
Real measurements only; writes results/runtime_benchmark.json. Run when CPU is free.
"""
import sys, json, time, tracemalloc
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
import logging; logging.basicConfig(level=logging.ERROR)

from data.loaders import DATASET_REGISTRY
from data.partitioner import partition_iid
from models.siamese import SiameseEncoder, train_siamese
from models.mlp import AnomalyAutoencoder
from dedup.local_dedup import LocalDeduplicator
from dedup.sil_gate import SILGate
from privacy.dp_trainer import DPTrainer
from privacy.accountant import compute_sigma_for_total_epsilon
from experiments.run_full_paper_sweep import DS_CFG, K
from sklearn.model_selection import train_test_split


def bench(ds):
    cfg = DS_CFG[ds]
    from experiments.cached_load import cached_load
    X, y = cached_load(ds, cfg["max_samples"])
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    enc = SiameseEncoder(Xtr.shape[1], cfg["siamese_emb"])
    enc = train_siamese(enc, Xtr, ytr, epochs=cfg["siamese_epochs"], window=cfg["siamese_window"])
    sil = SILGate(cfg["sil_sigma"]); sil.fit(Xtr[ytr == 0])
    p = partition_iid(Xtr, ytr, K, random_state=0)[0]
    B, E, R = cfg["batch_size"], cfg["local_epochs"], cfg["n_rounds"]

    # S0
    t = time.perf_counter(); _ = sil.flag(p.X); s0 = time.perf_counter() - t
    # S1 (encode + index + dedup) with peak memory
    tracemalloc.start(); t = time.perf_counter()
    dr = LocalDeduplicator(similarity_threshold=cfg["similarity_threshold"]).deduplicate(
        p.X, p.y, encoder=enc, sil_gate=sil)
    s1 = time.perf_counter() - t; mem = tracemalloc.get_traced_memory()[1] / 1e6; tracemalloc.stop()
    Uk = dr.X_dedup
    # S2: one DP-SGD round on U_k
    n = len(Uk); bs = min(B, max(2, n - 1)); steps = max(1, n // bs) * E
    sig = compute_sigma_for_total_epsilon(1.0, n, bs, steps, R, cfg["delta"])
    model = AnomalyAutoencoder(Xtr.shape[1], cfg["bottleneck"])
    tr = DPTrainer(model, sigma=sig, max_grad_norm=1.0)
    t = time.perf_counter(); tr.fit(Uk, epochs=E, batch_size=bs, use_dp=True); s2 = time.perf_counter() - t
    n_params = sum(pp.numel() for pp in model.parameters())
    comm = 2 * n_params * 4  # bytes/round (up+down, float32); identical for C5/C6

    # scaling probe: S1 time vs n
    scale = {}
    for frac in [0.25, 0.5, 1.0]:
        m = max(64, int(frac * len(p.X))); sub = p.X[:m]; suby = p.y[:m]
        t = time.perf_counter()
        LocalDeduplicator(similarity_threshold=cfg["similarity_threshold"]).deduplicate(sub, suby, encoder=enc, sil_gate=sil)
        scale[m] = round(time.perf_counter() - t, 3)
    return {"dataset": ds, "n_client": int(len(p.X)), "U_k": int(n),
            "s0_sil_s": round(s0, 4), "s1_dedup_s": round(s1, 3), "s1_peak_mem_MB": round(mem, 1),
            "s2_dpsgd_round_s": round(s2, 3), "n_params": int(n_params),
            "comm_bytes_per_round": int(comm), "s1_scaling_time_by_n": scale}


if __name__ == "__main__":
    out = [bench(ds) for ds in ["skab", "swat", "tep"]]
    (ROOT / "results" / "runtime_benchmark.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
