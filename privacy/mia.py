"""
Membership Inference Attack (MIA) evaluator — Shokri et al. S&P 2017 style.

Methodology
-----------
1. Split the training data into "member" (actually used to train) and
   "non-member" (held-out) sets.
2. Query the target model for reconstruction error on both sets.
3. Train a logistic-regression attack model on (loss, label=member/non-member).
4. Report AUC and accuracy of the attack.

Interpretation: AUC ≈ 0.5 → no membership signal detectable; AUC → 1.0 → strong leak.
The member/non-member split is decisive and confounds in BOTH directions: a contiguous
temporal split inflates the AUC (process drift), while a uniform-random split deflates it
on autocorrelated streams (near-duplicate non-members). The reported evaluation uses a
blocked split with a guard gap, validated by a positive control that a working attack
must expose (see experiments/mia_privacy_final.py).
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score

log = logging.getLogger(__name__)


class MIAEvaluator:
    """
    Lightweight MIA based on reconstruction-loss gap between members and
    non-members.

    Parameters
    ----------
    n_shadow_samples : int
        Max samples drawn from member/non-member pools for the attack.
    """

    def __init__(self, n_shadow_samples: int = 2000):
        self.n_shadow_samples = n_shadow_samples
        self._attack_model    = None

    def evaluate(
        self,
        target_model,         # AnomalyAutoencoder with .anomaly_score()
        X_member:    np.ndarray,
        X_nonmember: np.ndarray,
        seed:        int = 42,
    ) -> dict:
        """
        Run the MIA and return metrics.

        Parameters
        ----------
        target_model : trained AnomalyAutoencoder
        X_member     : data the model WAS trained on
        X_nonmember  : held-out data the model was NOT trained on
        """
        rng = np.random.default_rng(seed)

        n = min(self.n_shadow_samples, len(X_member), len(X_nonmember))
        idx_m  = rng.choice(len(X_member),    size=n, replace=False)
        idx_nm = rng.choice(len(X_nonmember), size=n, replace=False)

        X_m  = X_member[idx_m]
        X_nm = X_nonmember[idx_nm]

        # Feature: reconstruction error (lower for members in overfit models)
        loss_m  = target_model.anomaly_score(X_m).reshape(-1, 1)
        loss_nm = target_model.anomaly_score(X_nm).reshape(-1, 1)

        X_attack = np.vstack([loss_m, loss_nm]).astype(np.float32)
        y_attack = np.concatenate([np.ones(n), np.zeros(n)]).astype(int)

        # Attack model: logistic regression (deliberately weak so we measure
        # the signal from the model, not from the attack architecture)
        clf = LogisticRegression(max_iter=500, random_state=seed)
        scores = cross_val_score(clf, X_attack, y_attack,
                                 cv=5, scoring="roc_auc")

        clf.fit(X_attack, y_attack)
        y_pred = clf.predict(X_attack)
        auc    = float(roc_auc_score(y_attack, clf.predict_proba(X_attack)[:, 1]))
        acc    = float(accuracy_score(y_attack, y_pred))

        # Privacy advantage = AUC - 0.5 (0 = no leakage)
        adv = auc - 0.5

        result = {
            "mia_auc":         round(auc, 4),
            "mia_accuracy":    round(acc, 4),
            "mia_advantage":   round(adv, 4),
            "mia_auc_cv_mean": round(float(scores.mean()), 4),
            "mia_auc_cv_std":  round(float(scores.std()), 4),
            "n_member":        n,
            "n_nonmember":     n,
        }
        log.info("MIA: AUC=%.4f  Acc=%.4f  Advantage=%.4f", auc, acc, adv)
        return result
