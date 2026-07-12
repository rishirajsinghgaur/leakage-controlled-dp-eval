"""
DP-SGD trainer using Opacus — wraps PrivacyEngine for DP-FL anomaly detection clients.

The novel coupling lives here: when dedup=True, the engine is attached to
the *deduplicated* dataset U_k, so the accountant tracks privacy over
|U_k| samples (smaller → better amplification at matched σ).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)


class DPTrainer:
    """
    Trains an AnomalyAutoencoder with DP-SGD (Opacus).

    Usage
    -----
    trainer = DPTrainer(model, sigma=1.2, max_grad_norm=1.0)
    trainer.fit(X_train, epochs=5, batch_size=64)
    epsilon_spent = trainer.epsilon_spent(delta=1e-5)
    weights       = trainer.get_weights()
    """

    def __init__(
        self,
        model:          nn.Module,
        sigma:          float = 1.0,
        max_grad_norm:  float = 1.0,
        device:         str   = "cpu",
    ):
        self.model         = model.to(device)
        self.sigma         = sigma
        self.max_grad_norm = max_grad_norm
        self.device        = device
        self._engine       = None
        self._dp_model     = None
        self._dp_opt       = None
        self._dp_loader    = None

    def fit(
        self,
        X: np.ndarray,
        epochs: int       = 5,
        batch_size: int   = 64,
        lr: float         = 1e-3,
        use_dp: bool      = True,
    ) -> float:
        """
        Train for `epochs` passes.  Returns reconstruction loss.
        When use_dp=False, trains without Opacus (used for non-DP baselines).
        """
        dataset = TensorDataset(torch.from_numpy(X).float())
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn   = nn.MSELoss()

        if use_dp:
            try:
                from opacus import PrivacyEngine
            except ImportError:
                raise ImportError("Install opacus: pip install opacus")

            engine = PrivacyEngine()
            self.model, optimizer, loader = engine.make_private(
                module=self.model,
                optimizer=optimizer,
                data_loader=loader,
                noise_multiplier=self.sigma,
                max_grad_norm=self.max_grad_norm,
            )
            self._engine = engine
        else:
            self._engine = None

        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon = self.model(batch)
                loss  = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1

            total_loss += epoch_loss
            if (epoch + 1) % max(1, epochs // 3) == 0:
                log.debug("Epoch %d/%d  loss=%.5f", epoch + 1, epochs,
                          epoch_loss / max(1, len(loader)))

        return total_loss / max(1, n_batches)

    def epsilon_spent(self, delta: float = 1e-5) -> float:
        """Return cumulative (ε, δ) privacy expenditure."""
        if self._engine is None:
            return float("inf")   # non-DP run
        try:
            return float(self._engine.get_epsilon(delta=delta))
        except Exception as e:
            log.warning("Could not retrieve ε from engine: %s", e)
            return float("nan")

    def get_weights(self) -> list:
        """Return trainable weights as numpy arrays (for Flower aggregation)."""
        # Unwrap Opacus GradSampleModule if needed
        m = getattr(self.model, "_module", self.model)
        return [v.detach().cpu().numpy() for v in m.state_dict().values()]

    def set_weights(self, weights: list) -> None:
        """Load weights from server (Flower round start)."""
        m = getattr(self.model, "_module", self.model)
        state = {k: torch.from_numpy(v) for k, v in
                 zip(m.state_dict().keys(), weights)}
        m.load_state_dict(state, strict=True)
