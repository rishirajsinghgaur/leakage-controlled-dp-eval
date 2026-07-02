"""
Federated aggregation strategies for DP-FL anomaly detection.

FedAvg  — McMahan et al. AISTATS 2017
FedProx — Li et al. MLSys 2020  (adds proximal term μ‖w-w_global‖² to local loss)

Both are implemented here so the FL client can use either without
depending on flwr strategy internals.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import List

import numpy as np


class Strategy(str, Enum):
    FEDAVG  = "fedavg"
    FEDPROX = "fedprox"


def aggregate_fedavg(
    weights_list: List[List[np.ndarray]],
    n_samples:    List[int],
) -> List[np.ndarray]:
    """
    Weighted average of client model weights, weighted by n_samples.
    Returns averaged weights (same structure as input).
    """
    total = sum(n_samples)
    agg   = [np.zeros_like(w) for w in weights_list[0]]

    for weights, n in zip(weights_list, n_samples):
        for i, layer in enumerate(weights):
            agg[i] += layer * (n / total)

    return agg


def fedprox_proximal_term(
    local_params:  List[np.ndarray],
    global_params: List[np.ndarray],
    mu: float = 0.01,
) -> float:
    """
    Compute the FedProx proximal regularisation term:
        (μ/2) * ‖w_local - w_global‖²

    Returned as a scalar float — added to the loss in the client
    training loop when strategy == "fedprox".
    """
    diff = sum(
        np.sum((lp - gp) ** 2)
        for lp, gp in zip(local_params, global_params)
    )
    return (mu / 2.0) * float(diff)
