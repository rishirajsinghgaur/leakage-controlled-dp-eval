"""
IID and Dirichlet non-IID partitioning for DP-FL anomaly detection.

ClientPartition is a lightweight dataclass holding one client's slice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ClientPartition:
    client_id: int
    X: np.ndarray
    y: np.ndarray
    # optional: indices into the original dataset (for reproducibility checks)
    indices: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def n_samples(self) -> int:
        return len(self.X)

    @property
    def anomaly_rate(self) -> float:
        return float(self.y.mean()) if len(self.y) else 0.0

    def __repr__(self) -> str:
        return (f"ClientPartition(id={self.client_id}, n={self.n_samples}, "
                f"anom={self.anomaly_rate:.2%})")


# ─────────────────────────────────────────────────────────────────────────────
# IID partition
# ─────────────────────────────────────────────────────────────────────────────

def partition_iid(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    random_state: int = 42,
) -> List[ClientPartition]:
    """
    Randomly shuffle and split data into num_clients equal-ish shards.
    Each client gets approximately the same class distribution (IID).
    """
    rng = np.random.default_rng(random_state)
    n = len(X)
    indices = rng.permutation(n)
    shards = np.array_split(indices, num_clients)

    partitions = []
    for cid, shard in enumerate(shards):
        partitions.append(ClientPartition(
            client_id=cid,
            X=X[shard],
            y=y[shard],
            indices=shard,
        ))
        log.debug("IID client %d: n=%d, anom=%.2f%%", cid, len(shard),
                  100 * y[shard].mean())

    _log_partition_stats("IID", partitions)
    return partitions


# ─────────────────────────────────────────────────────────────────────────────
# Dirichlet non-IID partition
# ─────────────────────────────────────────────────────────────────────────────

def partition_dirichlet(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    random_state: int = 42,
    min_samples: int = 50,
) -> List[ClientPartition]:
    """
    Partition data with label-distribution heterogeneity via Dirichlet(α).

    Small α → highly heterogeneous (clients may see only one class).
    Large α → approaches IID.

    For binary y:  label_proportions drawn from Dir(α) over {0,1}.
    For multi-class y: standard LDA-style split per class.
    """
    rng = np.random.default_rng(random_state)
    classes = np.unique(y)
    n_classes = len(classes)

    # Group indices by class
    class_indices = {c: np.where(y == c)[0] for c in classes}
    for c in classes:
        rng.shuffle(class_indices[c])

    # Draw proportions: shape (num_clients, n_classes)
    proportions = rng.dirichlet(alpha=np.full(num_clients, alpha), size=n_classes).T
    # proportions[k, c] = fraction of class c allocated to client k

    client_idx: List[List[int]] = [[] for _ in range(num_clients)]

    for c_idx, c in enumerate(classes):
        c_inds = class_indices[c]
        n_c = len(c_inds)
        splits = (np.cumsum(proportions[:, c_idx]) * n_c).astype(int)
        splits = np.clip(splits, 0, n_c)
        prev = 0
        for k in range(num_clients):
            end = splits[k] if k < num_clients - 1 else n_c
            client_idx[k].extend(c_inds[prev:end].tolist())
            prev = end

    partitions = []
    for cid, idx_list in enumerate(client_idx):
        idx = np.array(idx_list, dtype=int)
        if len(idx) < min_samples:
            log.warning("Client %d got only %d samples (α=%.2f). "
                        "Consider increasing min_samples or raising α.", cid, len(idx), alpha)
        rng.shuffle(idx)
        partitions.append(ClientPartition(
            client_id=cid,
            X=X[idx],
            y=y[idx],
            indices=idx,
        ))

    _log_partition_stats(f"Dirichlet(α={alpha})", partitions)
    return partitions


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _log_partition_stats(scheme: str, partitions: List[ClientPartition]) -> None:
    sizes  = [p.n_samples for p in partitions]
    anoms  = [p.anomaly_rate for p in partitions]
    log.info(
        "%s partition: K=%d | sizes min=%d max=%d mean=%.0f | "
        "anom_rate min=%.2f%% max=%.2f%%",
        scheme, len(partitions),
        min(sizes), max(sizes), np.mean(sizes),
        100 * min(anoms), 100 * max(anoms),
    )


def verify_non_iid(partitions: List[ClientPartition]) -> dict:
    """Return statistics useful for verifying non-IID-ness."""
    rates = np.array([p.anomaly_rate for p in partitions])
    return {
        "anomaly_rate_std": float(rates.std()),
        "anomaly_rate_min": float(rates.min()),
        "anomaly_rate_max": float(rates.max()),
        "anomaly_rate_range": float(rates.max() - rates.min()),
        "sizes": [p.n_samples for p in partitions],
    }
