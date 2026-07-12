"""
DP-FL anomaly detection local client.

Plain Python class — no Flower actor dependency.
Works with the manual simulation loop in fl/server.py.

Each client implements S0-S2:
  [S0] SIL gate    — hard-preserve safety-critical samples
  [S1] Local dedup — near-duplicate removal via Siamese encoder → U_k
  [S2] DP-SGD      — Opacus training over U_k

The novel coupling (S1→S2): the Opacus PrivacyEngine is attached to U_k,
so the privacy accountant runs over |U_k| < |X_k| samples.
At a *matched* target ε, σ is calibrated to |U_k|: smaller dataset with
higher subsampling rate q=B/|U_k| requires slightly higher σ to maintain ε,
but the gradient diversity benefit (each step uses unique data) compensates.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np

from data.partitioner import ClientPartition
from models.mlp       import AnomalyAutoencoder, evaluate_anomaly_detector
from models.siamese   import SiameseEncoder
from dedup.sil_gate   import SILGate
from dedup.local_dedup import LocalDeduplicator, DedupResult
from privacy.dp_trainer import DPTrainer
from privacy.accountant import compute_sigma_for_epsilon, compute_sigma_for_total_epsilon

log = logging.getLogger(__name__)


class LocalClient:
    """
    One federated client that runs locally inside the simulation loop.

    Parameters
    ----------
    partition        : this client's data slice
    input_dim        : number of sensor features
    config           : experiment config dict
    dedup_enabled    : run S1 (local dedup) before S2 (DP-SGD)
    dp_enabled       : use DP-SGD (S2)
    target_epsilon   : calibrate σ to achieve this ε
    sigma_override   : use this σ directly (skips calibration)
    siamese_encoder  : pre-trained shared encoder for similarity search
    global_train_X   : full training set (used to fit SIL gate properly)
    proximal_mu      : FedProx μ (0 = FedAvg)
    """

    def __init__(
        self,
        client_id:       int,
        partition:       ClientPartition,
        input_dim:       int,
        config:          dict,
        dedup_enabled:   bool = True,
        dp_enabled:      bool = True,
        target_epsilon:  float = 1.0,
        sigma_override:  Optional[float] = None,
        siamese_encoder: Optional[SiameseEncoder] = None,
        global_train_X:  Optional[np.ndarray] = None,
        proximal_mu:     float = 0.0,
    ):
        self.cid           = client_id
        self.partition     = partition
        self.input_dim     = input_dim
        self.cfg           = config
        self.dedup_enabled = dedup_enabled
        self.dp_enabled    = dp_enabled
        self.target_eps    = target_epsilon
        self.sigma_override = sigma_override
        self.encoder       = siamese_encoder
        self.proximal_mu   = proximal_mu

        # SIL gate — fit on the global training distribution
        self.sil_gate = SILGate(sigma_threshold=config.get("sil_sigma", 3.0))
        self.sil_gate.fit(global_train_X if global_train_X is not None else partition.X)

        self.deduplicator = LocalDeduplicator(
            similarity_threshold=config.get("similarity_threshold", 0.97),
        )

        # Dedup result cached after first computation (encoder is frozen)
        self._dedup_result: Optional[DedupResult] = None

        # Sigma cached after first calibration (n_tr and batch_size are constant)
        self._sigma_cache: Optional[float] = None

        # Fresh model each construction; weights set from server before fit()
        self.model = AnomalyAutoencoder(
            input_dim  = input_dim,
            bottleneck = config.get("bottleneck", 8),
        )

    # ── API expected by fl/server.py ──────────────────────────────────────────

    def fit(
        self,
        global_weights: List[np.ndarray],
        config: dict,
    ) -> Tuple[List[np.ndarray], int, dict]:
        """
        One local training round.

        Returns
        -------
        weights  : updated model weights
        n        : number of training samples used (|U_k| or |X_k|)
        metrics  : dict of scalars for aggregation/logging
        """
        # Always create a fresh model each round — avoids Opacus double-hook error
        # (PrivacyEngine.make_private() cannot be called twice on the same model)
        fresh_model = AnomalyAutoencoder(
            input_dim  = self.input_dim,
            bottleneck = self.cfg.get("bottleneck", 8),
        )
        fresh_model.set_weights(global_weights)
        self.model = fresh_model

        X_local = self.partition.X
        y_local = self.partition.y

        # Corrected method: a reconstruction detector must be trained on
        # NORMAL data only; optionally remove temporal redundancy from that normal
        # stream. These take precedence over the legacy mixed-data dedup path.
        if self.cfg.get("train_normal_only", False):
            if self._dedup_result is None:
                mask = (y_local == 0)
                X_base = X_local[mask] if mask.sum() >= 2 else X_local
                # selection_mode: full | tdedup | random | fps (matched to tdedup budget).
                # Back-compat: temporal_dedup=True maps to "tdedup".
                mode = self.cfg.get("selection_mode",
                                    "tdedup" if self.cfg.get("temporal_dedup", False) else "full")
                if mode != "full" and self.encoder is not None and len(X_base) > 4:
                    from experiments.principled_method import seq_temporal_dedup
                    kept, _tau = seq_temporal_dedup(X_base, self.encoder, keep_quantile=0.5)
                    budget = max(2, len(kept))
                    if mode == "tdedup":
                        X_base = X_base[kept]
                    elif mode == "random":
                        rng = np.random.default_rng(1000 + int(self.cid))
                        sel = np.sort(rng.choice(len(X_base), size=min(budget, len(X_base)), replace=False))
                        X_base = X_base[sel]
                    elif mode == "fps":
                        E = self.encoder.encode(X_base)
                        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
                        b = min(budget, len(E)); selj = [0]; d = 1.0 - E @ E[0]
                        for _ in range(1, b):
                            j = int(np.argmax(d)); selj.append(j); d = np.minimum(d, 1.0 - E @ E[j])
                        X_base = X_base[np.array(sorted(selj))]
                self._dedup_result = X_base                # cache the corrected training set
            X_tr = self._dedup_result
            rho  = len(X_tr) / max(1, len(X_local))
        elif self.dedup_enabled:
            # Legacy (deprecated) mixed-data near-duplicate removal path.
            if self._dedup_result is None:
                self._dedup_result = self.deduplicator.deduplicate(
                    X_local, y_local,
                    encoder  = self.encoder,
                    sil_gate = self.sil_gate,
                )
            dr   = self._dedup_result
            X_tr = dr.X_dedup
            rho  = dr.retention_rho
        else:
            X_tr = X_local
            rho  = 1.0

        n_tr = len(X_tr)

        if n_tr < 2:
            log.warning("Client %d: fewer than 2 training samples; skipping round.", self.cid)
            return global_weights, 0, {"epsilon": 0.0, "sigma": 0.0, "rho": rho,
                                       "train_loss": 0.0, "n_dp": 0,
                                       "n_original": len(X_local), "client_id": self.cid}

        # S2: compute σ to hit target ε over |U_k|
        epochs     = config.get("local_epochs", self.cfg.get("local_epochs", 5))
        batch_size = min(config.get("batch_size", self.cfg.get("batch_size", 64)), n_tr - 1)
        n_steps    = max(1, (n_tr // batch_size)) * epochs

        if self.dp_enabled:
            if self.sigma_override is not None:
                sigma = self.sigma_override
            elif self._sigma_cache is not None:
                sigma = self._sigma_cache
            else:
                n_rounds = self.cfg.get("n_rounds", 15)
                try:
                    # Calibrate σ for TOTAL training (all FL rounds) so that
                    # the end-to-end guarantee is (target_ε, δ)-DP.
                    sigma = compute_sigma_for_total_epsilon(
                        target_epsilon     = self.target_eps,
                        n_samples          = n_tr,
                        batch_size         = batch_size,
                        n_steps_per_round  = n_steps,
                        n_rounds           = n_rounds,
                        delta              = self.cfg.get("delta", 1e-5),
                    )
                    self._sigma_cache = sigma
                except ValueError as e:
                    log.warning("Client %d: σ calibration failed (%s) — using σ=4.0", self.cid, e)
                    sigma = 4.0
                    self._sigma_cache = sigma
        else:
            sigma = 0.0

        trainer = DPTrainer(
            model         = self.model,
            sigma         = sigma,
            max_grad_norm = self.cfg.get("max_grad_norm", 1.0),
            device        = "cpu",
        )

        if self.proximal_mu > 0:
            loss = _fit_fedprox(
                trainer, X_tr, epochs, batch_size,
                self.cfg.get("lr", 1e-3),
                self.proximal_mu, global_weights,
                use_dp=self.dp_enabled,
            )
        else:
            loss = trainer.fit(
                X_tr,
                epochs     = epochs,
                batch_size = batch_size,
                lr         = self.cfg.get("lr", 1e-3),
                use_dp     = self.dp_enabled,
            )

        eps_spent = trainer.epsilon_spent(delta=self.cfg.get("delta", 1e-5))
        weights   = trainer.get_weights()

        log.debug("Client %d  loss=%.5f ε=%.3f σ=%.3f ρ=%.3f n=%d",
                  self.cid, loss, eps_spent, sigma, rho, n_tr)

        metrics = {
            "train_loss": float(loss),
            "epsilon":    float(eps_spent),
            "sigma":      float(sigma),
            "rho":        float(rho),
            "n_dp":       int(n_tr),
            "n_original": int(len(X_local)),
            "client_id":  self.cid,
            "dedup":      int(self.dedup_enabled),
        }
        return weights, n_tr, metrics

    def evaluate(
        self,
        global_weights: List[np.ndarray],
        config: dict,
    ) -> Tuple[float, int, dict]:
        self.model.set_weights(global_weights)
        met  = evaluate_anomaly_detector(
            self.model, self.partition.X, self.partition.y,
            percentile=config.get("eval_percentile", 95.0),
        )
        loss = 1.0 - met["f1"]
        return loss, len(self.partition.X), met


# ─────────────────────────────────────────────────────────────────────────────
# Builder: factory used by run_simulation()
# ─────────────────────────────────────────────────────────────────────────────

def build_client_fn(
    partitions:      List[ClientPartition],
    input_dim:       int,
    config:          dict,
    dedup_enabled:   bool  = True,
    dp_enabled:      bool  = True,
    target_epsilon:  float = 1.0,
    sigma_override:  Optional[float] = None,
    siamese_encoder: Optional[SiameseEncoder] = None,
    global_train_X:  Optional[np.ndarray] = None,
    proximal_mu:     float = 0.0,
) -> Callable[[int], LocalClient]:
    """Return a client_fn(cid: int) → LocalClient."""

    def client_fn(cid: int) -> LocalClient:
        return LocalClient(
            client_id       = cid,
            partition       = partitions[cid],
            input_dim       = input_dim,
            config          = config,
            dedup_enabled   = dedup_enabled,
            dp_enabled      = dp_enabled,
            target_epsilon  = target_epsilon,
            sigma_override  = sigma_override,
            siamese_encoder = siamese_encoder,
            global_train_X  = global_train_X,
            proximal_mu     = proximal_mu,
        )

    return client_fn


# ─────────────────────────────────────────────────────────────────────────────
# FedProx + DP training
# ─────────────────────────────────────────────────────────────────────────────

def _fit_fedprox(
    trainer:        DPTrainer,
    X:              np.ndarray,
    epochs:         int,
    batch_size:     int,
    lr:             float,
    mu:             float,
    global_weights: List[np.ndarray],
    use_dp:         bool = True,
) -> float:
    """Train with FedProx proximal term, optionally with DP-SGD."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    model  = trainer.model
    device = trainer.device

    global_tensors = [torch.from_numpy(w).float().to(device) for w in global_weights]

    dataset = TensorDataset(torch.from_numpy(X).float())
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    if use_dp and trainer.sigma > 0:
        from opacus import PrivacyEngine
        engine = PrivacyEngine()
        model, opt, loader = engine.make_private(
            module=model, optimizer=opt, data_loader=loader,
            noise_multiplier=trainer.sigma, max_grad_norm=trainer.max_grad_norm,
        )
        trainer._engine = engine

    model.train()
    total_loss, n_b = 0.0, 0

    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon = model(batch)
            loss  = loss_fn(recon, batch)

            m = getattr(model, "_module", model)
            prox = sum(torch.sum((p - g) ** 2)
                       for p, g in zip(m.parameters(), global_tensors))
            loss = loss + (mu / 2.0) * prox

            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_b        += 1

    trainer.model = model
    return total_loss / max(1, n_b)
