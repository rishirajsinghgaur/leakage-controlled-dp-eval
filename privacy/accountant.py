"""
RDP / (ε,δ)-DP accounting utilities for DP-FL anomaly detection.

Wraps Opacus's built-in accountant so the rest of the codebase
has a simple, stable interface.

Correct framing of the dedup-DP coupling:
  When training on U_k (the deduplicated set, |U_k| = ρ|X_k|), the
  subsampling rate q = B/|U_k| is LARGER than B/|X_k|, and the number
  of steps T_k = ⌈|U_k|/B⌉ · E is proportionally SMALLER.

  Numerically, ε(U_k, same σ) > ε(X_k, same σ):
  deduplication does NOT reduce the privacy budget at fixed σ.

  The correct utility argument is INFORMATION DENSITY:
    At matched (ε, δ), both conditions spend the same budget.
    DP-FL anomaly detection trains on unique, non-redundant gradients from U_k,
    while DP-no-dedup wastes gradient steps on near-duplicate samples
    that provide approximately the same update direction.
    The result: better model utility per unit of privacy budget, even
    though σ is slightly larger for U_k.

compute_sigma_for_epsilon() binary-searches for the σ that achieves
exactly the target ε given (n_samples, batch_size, n_steps, delta).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class RDPRecord:
    """Summary of the privacy expenditure for one training run."""
    epsilon: float
    delta:   float
    sigma:   float
    n_steps: int
    sample_rate: float   # B / N
    n_samples:   int     # |U_k| (after dedup) or |X_k| (no dedup)


def compute_epsilon(
    n_samples:  int,
    batch_size: int,
    n_steps:    int,
    sigma:      float,
    delta:      float = 1e-5,
    accountant: str  = "rdp",
) -> float:
    """
    Compute the (ε, δ)-DP guarantee for DP-SGD training.

    Parameters
    ----------
    n_samples  : dataset size (|U_k| when dedup is on)
    batch_size : lot size
    n_steps    : total gradient steps across all FL rounds
    sigma      : noise multiplier
    delta      : DP delta
    accountant : "rdp" or "prv" (Opacus supports both)
    """
    try:
        from opacus.accountants import create_accountant
    except ImportError:
        raise ImportError("Install opacus: pip install opacus")

    sample_rate = batch_size / n_samples
    acc = create_accountant(mechanism="rdp")
    acc.history = [(sigma, sample_rate, n_steps)]
    eps = acc.get_epsilon(delta=delta)
    log.debug("ε=%.4f (σ=%.3f, q=%.4f, T=%d, δ=%s)", eps, sigma, sample_rate, n_steps, delta)
    return float(eps)


def compute_sigma_for_epsilon(
    target_epsilon: float,
    n_samples:      int,
    batch_size:     int,
    n_steps:        int,
    delta:          float = 1e-5,
    sigma_lo:       float = 0.1,
    sigma_hi:       float = 100.0,
    tol:            float = 1e-3,
) -> float:
    """
    Binary-search for the smallest σ that achieves ε ≤ target_epsilon.

    Returns σ (noise multiplier).
    """
    # Check feasibility: even at sigma_hi can we reach the target?
    eps_at_hi = compute_epsilon(n_samples, batch_size, n_steps, sigma_hi, delta)
    if eps_at_hi > target_epsilon:
        raise ValueError(
            f"Cannot reach ε={target_epsilon:.2f} even at σ={sigma_hi}. "
            f"Got ε={eps_at_hi:.3f}. Reduce n_steps or batch_size."
        )

    lo, hi = sigma_lo, sigma_hi
    for _ in range(64):    # 64 bisection steps → precision < 2^-64 range
        mid = (lo + hi) / 2.0
        eps = compute_epsilon(n_samples, batch_size, n_steps, mid, delta)
        if eps > target_epsilon:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < tol:
            break

    sigma = hi   # conservative: use the higher σ to stay within budget
    log.info("Target ε=%.2f → σ=%.4f (n=%d, B=%d, T=%d, δ=%s)",
             target_epsilon, sigma, n_samples, batch_size, n_steps, delta)
    return float(sigma)


def compute_total_epsilon(
    n_samples:  int,
    batch_size: int,
    n_steps_per_round: int,
    n_rounds:   int,
    sigma:      float,
    delta:      float = 1e-5,
) -> float:
    """
    Compute the TOTAL (ε, δ)-DP guarantee after n_rounds FL rounds.

    Each round runs n_steps_per_round gradient steps. The total privacy
    expenditure is computed by the RDP accountant for T_total = n_rounds *
    n_steps_per_round steps — the correct end-to-end guarantee.

    Note: σ calibrated with compute_sigma_for_epsilon(n_steps=n_steps_per_round)
    will achieve ε_per_round << ε_total. To achieve ε_total = target_epsilon,
    use compute_sigma_for_epsilon(n_steps=n_steps_total).
    """
    n_steps_total = n_steps_per_round * n_rounds
    return compute_epsilon(n_samples, batch_size, n_steps_total, sigma, delta)


def compute_sigma_for_total_epsilon(
    target_epsilon: float,
    n_samples:      int,
    batch_size:     int,
    n_steps_per_round: int,
    n_rounds:       int,
    delta:          float = 1e-5,
) -> float:
    """
    Calibrate σ so that the TOTAL training achieves ε ≤ target_epsilon.

    Correct for FL: calibrates over T_total = n_steps_per_round * n_rounds
    gradient steps, giving a valid end-to-end (ε, δ)-DP guarantee.
    """
    n_steps_total = n_steps_per_round * n_rounds
    return compute_sigma_for_epsilon(
        target_epsilon=target_epsilon,
        n_samples=n_samples,
        batch_size=batch_size,
        n_steps=n_steps_total,
        delta=delta,
    )


def privacy_report(
    n_original: int,
    n_deduped:  int,
    batch_size: int,
    n_steps:    int,
    sigma:      float,
    delta:      float = 1e-5,
) -> dict:
    """
    Compare privacy and utility properties of dedup vs no-dedup at matched ε.

    Key insight: at matched (ε, δ), σ_dedup > σ_nodedup because the smaller
    deduplicated set has higher per-step privacy cost (q = B/n is larger).
    The utility benefit comes from INFORMATION DENSITY: gradients from unique,
    non-redundant samples (U_k) are more informative than the same number of
    gradient steps on a dataset with near-duplicates.

    sigma_dedup    : noise multiplier calibrated for U_k (input to this fn)
    sigma_nodedup  : noise multiplier required to achieve same ε on X_k
    sigma_ratio    : sigma_dedup / sigma_nodedup  (> 1; dedup needs more noise)
    utility_gap    : qualitative — positive if data diversity dominates noise
    """
    rho = n_deduped / n_original

    # Sigma required for no-dedup at same eps budget
    # n_steps_nodedup scales proportionally with data size
    n_steps_nodedup = int(n_steps / rho) if rho > 0 else n_steps
    try:
        eps_at_matched_sigma = compute_epsilon(n_deduped, batch_size, n_steps, sigma, delta)
        sigma_nodedup = compute_sigma_for_epsilon(
            target_epsilon=eps_at_matched_sigma,
            n_samples=n_original, batch_size=batch_size,
            n_steps=n_steps_nodedup, delta=delta,
        )
        sigma_ratio = sigma / sigma_nodedup  # > 1 means dedup needs more noise
    except Exception:
        sigma_nodedup = 0.0
        sigma_ratio   = float("nan")

    return {
        "n_original":     n_original,
        "n_deduped":      n_deduped,
        "rho":            round(rho, 4),
        "sigma_dedup":    round(sigma, 4),
        "sigma_nodedup":  round(sigma_nodedup, 4),
        "sigma_ratio":    round(sigma_ratio, 4),  # > 1: dedup needs more noise at same eps
        "n_steps_dedup":  n_steps,
        "n_steps_nodedup": n_steps_nodedup,
        "step_reduction": round(1 - rho, 4),      # fraction of steps saved
        "delta":          delta,
        "batch_size":     batch_size,
    }
