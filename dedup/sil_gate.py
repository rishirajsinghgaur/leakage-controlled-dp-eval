"""
SIL Gate (S0) — hard safety constraint.

Samples flagged as safety-critical (SIL ≥ 2) are NEVER removed by the
deduplicator.  This is a one-line hard guarantee, not the paper's
contribution, but it must be present for the IEC-61508 coverage.

Detection heuristic: a sample is flagged when at least one feature
deviates more than `sigma_threshold` standard deviations from the
training-set mean.  A full SIL classification would require domain
knowledge; this statistical proxy is sufficient for the experiment.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SILGate:
    sigma_threshold: float = 3.0   # per-feature 3-sigma exceedance rule

    # Set during fit()
    _means: np.ndarray = None
    _stds:  np.ndarray = None

    def fit(self, X_train: np.ndarray) -> "SILGate":
        """Compute training statistics (must be called on training data only)."""
        self._means = X_train.mean(axis=0)
        self._stds  = X_train.std(axis=0) + 1e-10
        return self

    def flag(self, X: np.ndarray) -> np.ndarray:
        """
        Return boolean mask: True = safety-critical (must be preserved).

        A sample is flagged if any feature is outside
        [mean ± sigma_threshold * std] on the training distribution.
        """
        if self._means is None:
            raise RuntimeError("SILGate.fit() must be called before flag()")
        z = np.abs((X - self._means) / self._stds)
        return (z > self.sigma_threshold).any(axis=1)
