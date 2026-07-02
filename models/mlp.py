"""
Anomaly-detection autoencoder for DP-FL anomaly detection.

A shallow MLP autoencoder is chosen because:
  1. Small enough for stable DP-SGD on CPU (few parameters → manageable
     per-sample gradient tensors for Opacus).
  2. Reconstruction-error anomaly score is simple and universally applicable
     across TEP, SWaT, SKAB without task-specific heads.
  3. A symmetric 3-layer MLP encoder/decoder keeps the parameter count low.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class AnomalyAutoencoder(nn.Module):
    """
    MLP Autoencoder for anomaly detection via reconstruction error.

    Encoder: input_dim → h1 → h2 → bottleneck
    Decoder: bottleneck → h2 → h1 → input_dim

    Anomaly score = mean squared reconstruction error per sample.
    """

    def __init__(self, input_dim: int, bottleneck: int = 8):
        super().__init__()
        h1 = max(64, input_dim * 2)
        h2 = max(32, input_dim)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def anomaly_score(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Return per-sample MSE reconstruction error (higher = more anomalous)."""
        self.eval()
        device = next(self.parameters()).device
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X).float()),
            batch_size=batch_size,
            shuffle=False,
        )
        scores = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(device)
                recon = self.forward(batch)
                mse   = ((recon - batch) ** 2).mean(dim=1)
                scores.append(mse.cpu().numpy())
        return np.concatenate(scores)

    def get_weights(self) -> list:
        """Return model weights as a list of numpy arrays (Flower-compatible)."""
        return [v.cpu().numpy() for v in self.state_dict().values()]

    def set_weights(self, weights: list) -> None:
        """Load weights from a list of numpy arrays (Flower-compatible)."""
        state = {k: torch.from_numpy(v) for k, v in
                 zip(self.state_dict().keys(), weights)}
        self.load_state_dict(state, strict=True)


def evaluate_anomaly_detector(
    model: AnomalyAutoencoder,
    X: np.ndarray,
    y: np.ndarray,
    percentile: float = 95.0,
) -> dict:
    """
    Threshold anomaly scores at the given percentile and report
    F1, precision, recall, AUPRC.
    """
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        average_precision_score,
    )

    scores = model.anomaly_score(X)
    threshold = np.percentile(scores, percentile)
    y_pred = (scores >= threshold).astype(int)

    return {
        "f1":        float(f1_score(y, y_pred, zero_division=0)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall":    float(recall_score(y, y_pred, zero_division=0)),
        "auprc":     float(average_precision_score(y, scores)),
        "threshold": float(threshold),
        "anomaly_rate_pred": float(y_pred.mean()),
    }
