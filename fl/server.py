"""
DP-FL anomaly detection — Manual FL simulation server.

Implements FedAvg/FedProx aggregation in pure Python+NumPy without Ray
or Flower's heavyweight simulation backend.  This is intentional:
  - No Ray required (CPU-only, single-machine execution)
  - Deterministic and reproducible across seeds
  - Transparent for paper reproducibility
  - Correct Opacus integration (no actor isolation issues)

run_simulation() is the single entry-point for one experimental cell.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from models.mlp  import AnomalyAutoencoder, evaluate_anomaly_detector
from fl.strategies import aggregate_fedavg

log = logging.getLogger(__name__)


@dataclass
class ResultRecord:
    """Everything the sweep runner needs from one simulation cell."""
    dataset:    str
    condition:  str
    seed:       int
    final_f1:       float = 0.0
    final_recall:   float = 0.0
    final_auprc:    float = 0.0
    final_epsilon:  float = float("inf")
    final_sigma:    float = 0.0
    final_rho:      float = 1.0
    comm_bytes:     float = 0.0
    rounds_to_conv: int   = 0
    mia_auc:        float = float("nan")
    history: List[dict] = field(default_factory=list)
    extra:   dict       = field(default_factory=dict)
    final_weights: list = field(default_factory=list, repr=False)


def run_simulation(
    client_fn:      Callable,         # (cid: int) → LocalClient
    n_clients:      int,
    n_rounds:       int,
    input_dim:      int,
    config:         dict,
    X_test:         np.ndarray,
    y_test:         np.ndarray,
    condition_name: str  = "unnamed",
    dataset_name:   str  = "unknown",
    seed:           int  = 42,
    fraction_fit:   float = 1.0,
) -> ResultRecord:
    """
    Run one complete FL simulation using the manual aggregation loop.

    Parameters
    ----------
    client_fn : callable (cid: int) → LocalClient
    """
    rng    = np.random.default_rng(seed)
    n_sel  = max(1, int(n_clients * fraction_fit))

    # Initialise global model
    global_model = AnomalyAutoencoder(
        input_dim  = input_dim,
        bottleneck = config.get("bottleneck", 8),
    )
    global_weights = global_model.get_weights()

    history: List[dict] = []
    best_f1  = 0.0
    best_round = 1

    log.info("Starting FL simulation: %s | clients=%d rounds=%d",
             condition_name, n_clients, n_rounds)

    # Instantiate all clients once so dedup cache persists across rounds
    clients = {cid: client_fn(cid) for cid in range(n_clients)}

    # Early-stopping patience: stop if F1 hasn't improved by > 0.001 in 4 rounds
    _patience = config.get("patience", 4)
    _no_improve = 0

    for rnd in range(1, n_rounds + 1):
        # 1. Sample clients
        selected = rng.choice(n_clients, size=n_sel, replace=False).tolist()

        # 2. Local training on each selected client
        client_weights_list = []
        client_n_list       = []
        round_metrics       = []

        for cid in selected:
            client = clients[cid]
            w, n, metrics = client.fit(global_weights, config={
                "round": rnd,
                "local_epochs": config.get("local_epochs", 5),
                "batch_size":   config.get("batch_size",   64),
            })
            client_weights_list.append(w)
            client_n_list.append(n)
            round_metrics.append(metrics)

        # 3. Aggregate (FedAvg weighted by n_samples)
        global_weights = aggregate_fedavg(client_weights_list, client_n_list)

        # 4. Evaluate global model on test set
        global_model.set_weights(global_weights)
        test_met = evaluate_anomaly_detector(
            global_model, X_test, y_test,
            percentile=config.get("eval_percentile", 95.0),
        )

        # Track best round
        if test_met["f1"] > best_f1 + 0.001:
            best_f1    = test_met["f1"]
            best_round = rnd
            _no_improve = 0
        else:
            _no_improve += 1

        # Aggregate per-round client metrics (weighted mean)
        total_n = sum(client_n_list)
        def wavg(key):
            return sum(m.get(key, 0) * n / total_n
                       for m, n in zip(round_metrics, client_n_list))

        entry = {
            "round":       rnd,
            "f1":          test_met["f1"],
            "recall":      test_met["recall"],
            "auprc":       test_met["auprc"],
            "epsilon":     wavg("epsilon"),
            "sigma":       wavg("sigma"),
            "rho":         wavg("rho"),
            "train_loss":  wavg("train_loss"),
            "n_clients":   len(selected),
        }
        history.append(entry)
        log.info("Round %2d/%d  F1=%.4f AUPRC=%.4f ε=%.3f ρ=%.3f",
                 rnd, n_rounds,
                 entry["f1"], entry["auprc"], entry["epsilon"], entry["rho"])

        # Early stopping: terminate once converged
        if _no_improve >= _patience and rnd >= 6:
            log.info("Early stop at round %d (no F1 improvement in %d rounds).",
                     rnd, _patience)
            break

    # Final metrics from last round
    last = history[-1] if history else {}

    # Communication bytes: upload + download, all rounds, selected clients
    n_params   = sum(p.numel() for p in global_model.parameters())
    comm_bytes = 2.0 * n_rounds * n_sel * n_params * 4  # float32 = 4 bytes

    result = ResultRecord(
        dataset        = dataset_name,
        condition      = condition_name,
        seed           = seed,
        final_f1       = last.get("f1",     0.0),
        final_recall   = last.get("recall", 0.0),
        final_auprc    = last.get("auprc",  0.0),
        final_epsilon  = last.get("epsilon", float("inf")),
        final_sigma    = last.get("sigma",   0.0),
        final_rho      = last.get("rho",     1.0),
        comm_bytes     = comm_bytes,
        rounds_to_conv = best_round,
        history        = history,
        final_weights  = global_weights,
    )
    log.info("Simulation done: %s | F1=%.4f AUPRC=%.4f ε=%.3f ρ=%.3f best_round=%d",
             condition_name, result.final_f1, result.final_auprc,
             result.final_epsilon, result.final_rho, best_round)
    return result
