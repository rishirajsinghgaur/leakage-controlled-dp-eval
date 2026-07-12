"""
Local Deduplicator (S1) — the core of DP-FL anomaly detection.

Pipeline per client:
  1. SIL gate  → mark safety-critical samples (always kept)
  2. Siamese encoder → L2-normalised embeddings
  3. FAISS HNSW approximate nearest-neighbour search
  4. Threshold on cosine similarity → identify near-duplicate pairs
  5. From each duplicate cluster retain exactly one representative
  6. Union: safety-critical ∪ unique representatives = U_k

The retention fraction ρ = |U_k| / |X_k| is the dedup knob swept in
ablation (a) of the experiment matrix.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

try:
    import faiss
    _FAISS = True
except ImportError:
    _FAISS = False
    log.warning("faiss-cpu not found; falling back to brute-force dedup (slow on large N).")


@dataclass
class DedupResult:
    X_dedup:       np.ndarray            # retained samples
    y_dedup:       np.ndarray            # retained labels
    retain_mask:   np.ndarray            # bool mask into original X
    n_original:    int = 0
    n_retained:    int = 0
    n_sil_kept:    int = 0               # samples kept by SIL gate
    retention_rho: float = 1.0          # |U_k| / |X_k|


class LocalDeduplicator:
    """
    Applies near-duplicate removal to one client's local dataset.

    Parameters
    ----------
    similarity_threshold : float
        Cosine-similarity threshold above which two samples are
        considered near-duplicates.  Default 0.97 (tight for sensor data).
        Set to None to use adaptive_threshold() instead.
    adaptive : bool
        If True, ignore similarity_threshold and compute a per-client
        adaptive threshold τ_k = μ_k + 2·σ_k from the embedding
        similarity distribution.  Novel contribution: each client adapts
        to its own local redundancy level.
    hnsw_m : int
        FAISS HNSW connectivity parameter.
    max_candidates : int
        K in the KNN search (how many neighbours checked per query).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.97,
        adaptive: bool = False,
        hnsw_m: int = 32,
        max_candidates: int = 64,
    ):
        self.similarity_threshold = similarity_threshold
        self.adaptive             = adaptive
        self.hnsw_m               = hnsw_m
        self.max_candidates       = max_candidates

    def deduplicate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        encoder,          # SiameseEncoder (or None → use raw normalised features)
        sil_gate=None,    # SILGate (or None → no hard preserve)
    ) -> DedupResult:
        """
        Parameters
        ----------
        X        : (N, D) float32
        y        : (N,)   int
        encoder  : SiameseEncoder with .encode(X) → (N, E) float32
        sil_gate : SILGate with .flag(X) → (N,) bool
        """
        N = len(X)

        # S0: safety-critical mask (hard preserve)
        if sil_gate is not None:
            sil_mask = sil_gate.flag(X)
        else:
            sil_mask = np.zeros(N, dtype=bool)

        n_sil = int(sil_mask.sum())

        # Samples eligible for dedup = non-safety-critical
        eligible_idx = np.where(~sil_mask)[0]

        if len(eligible_idx) == 0:
            log.debug("All samples SIL-protected; no dedup performed.")
            return DedupResult(X, y, np.ones(N, dtype=bool), N, N, n_sil, 1.0)

        X_elig = X[eligible_idx]

        # Compute embeddings
        if encoder is not None:
            embeddings = encoder.encode(X_elig)              # already L2-normalised
        else:
            norms = np.linalg.norm(X_elig, axis=1, keepdims=True) + 1e-8
            embeddings = (X_elig / norms).astype(np.float32)

        # Adaptive or fixed threshold
        if self.adaptive:
            tau = _adaptive_threshold(embeddings)
            log.info("Adaptive τ_k = %.4f (client local embedding distribution)", tau)
        else:
            tau = self.similarity_threshold

        # Find duplicate pairs
        dup_set = _find_duplicates(
            embeddings,
            sim_threshold=tau,
            hnsw_m=self.hnsw_m,
            k=self.max_candidates,
        )

        # Union-Find: cluster all transitively-connected duplicates and
        # retain exactly one representative per cluster (lowest index).
        parent = list(range(len(eligible_idx)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]   # path compression
                x = parent[x]
            return x

        for (i, j) in dup_set:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri   # always keep lower index as root

        # Remove all non-root members of each cluster
        to_remove = {i for i in range(len(eligible_idx)) if find(i) != i}

        # Retained eligible indices (local to eligible_idx)
        kept_local = np.array([i for i in range(len(eligible_idx)) if i not in to_remove])

        # Map back to global indices
        kept_global_eligible = eligible_idx[kept_local] if len(kept_local) else np.array([], dtype=int)
        kept_global_sil      = np.where(sil_mask)[0]

        retained = np.sort(np.concatenate([kept_global_sil, kept_global_eligible]).astype(int))

        retain_mask = np.zeros(N, dtype=bool)
        retain_mask[retained] = True

        rho = len(retained) / N
        log.info(
            "Dedup: N=%d → U_k=%d (ρ=%.3f) | SIL-kept=%d | dups-removed=%d",
            N, len(retained), rho, n_sil, N - len(retained),
        )

        return DedupResult(
            X_dedup     = X[retain_mask],
            y_dedup     = y[retain_mask],
            retain_mask = retain_mask,
            n_original  = N,
            n_retained  = int(retain_mask.sum()),
            n_sil_kept  = n_sil,
            retention_rho = rho,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal: FAISS (or brute-force) duplicate pair finder
# ─────────────────────────────────────────────────────────────────────────────

def _adaptive_threshold(embeddings: np.ndarray, sample_size: int = 2000) -> float:
    """
    Compute a per-client adaptive similarity threshold τ_k.

    τ_k = μ_k + 2·σ_k  where μ, σ are the mean and std of pairwise cosine
    similarities sampled from the client's embedding set.  This selects the
    top ~2.5% of the similarity distribution as the near-duplicate region,
    adapting automatically to each client's local redundancy level.

    Complexity: O(sample_size²) — cheap for sample_size ≤ 2000.
    """
    n = len(embeddings)
    if n <= 1:
        return 0.97

    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    emb = embeddings[idx].astype(np.float32)
    # Cosine similarity matrix (embeddings already L2-normed)
    sim = emb @ emb.T
    upper = sim[np.triu_indices(len(emb), k=1)]
    mu, sigma = float(np.mean(upper)), float(np.std(upper))
    # Clip to [0.70, 0.995] to avoid degenerate extremes
    tau = float(np.clip(mu + 2.0 * sigma, 0.70, 0.995))
    return tau


def _find_duplicates(
    embeddings: np.ndarray,
    sim_threshold: float,
    hnsw_m: int,
    k: int,
) -> set:
    """
    Return set of (i, j) pairs where cosine_sim(emb[i], emb[j]) ≥ threshold,
    i < j.
    """
    n, d = embeddings.shape
    emb  = embeddings.astype(np.float32)

    dup_pairs = set()

    if _FAISS and n > 10:
        # FAISS HNSW with inner-product metric (embeddings already L2-normed,
        # so inner product = cosine similarity).
        index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch       = min(100, n)
        index.add(emb)

        k_search = min(k + 1, n)   # +1 because the point itself is returned
        sims, neighbours = index.search(emb, k_search)

        for i in range(n):
            for rank in range(1, k_search):  # skip rank-0 (self)
                j   = int(neighbours[i, rank])
                sim = float(sims[i, rank])
                if j < 0:
                    break
                if sim >= sim_threshold and j > i:
                    dup_pairs.add((i, j))
    else:
        # Brute-force fallback
        sims = emb @ emb.T
        for i in range(n):
            for j in range(i + 1, min(i + k + 1, n)):
                if sims[i, j] >= sim_threshold:
                    dup_pairs.add((i, j))

    return dup_pairs
