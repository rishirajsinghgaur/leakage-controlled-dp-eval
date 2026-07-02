"""
Siamese encoder for near-duplicate detection in DP-FL anomaly detection.

Ported and cleaned from capclone/improved_uncertainty_aware_framework.py.
Used in the local dedup stage (S1) — NOT sent to the server.
Kept small for CPU inference: input_dim → 128 → 64 → embedding_dim.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class SiameseEncoder(nn.Module):
    """
    Triplet-loss Siamese encoder.

    Produces L2-normalised embeddings used for cosine-similarity
    near-duplicate detection.
    """

    def __init__(self, input_dim: int, embedding_dim: int = 32):
        super().__init__()
        self.input_dim    = input_dim
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        return F.normalize(emb, dim=-1)

    def encode(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Encode numpy array → L2-normalised embeddings (numpy)."""
        self.eval()
        device = next(self.parameters()).device
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X).float()),
            batch_size=batch_size,
            shuffle=False,
        )
        chunks = []
        with torch.no_grad():
            for (batch,) in loader:
                chunks.append(self.forward(batch.to(device)).cpu().numpy())
        return np.vstack(chunks)


class TripletDataset(torch.utils.data.Dataset):
    """
    Generates triplets (anchor, positive, negative) from NORMAL (public) IoT data.

    Privacy-safe design: trained ONLY on the normal-operation public subset
    (y=0 samples), using temporal proximity as the similarity signal.
    Anchor and positive are consecutive-window pairs (similar sensor states);
    negative is a sample drawn uniformly from the normal set (likely different).

    This avoids the need for anomaly labels and avoids exposing private
    client data to the server during encoder pre-training.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray = None,
                 window: int = 5, random_state: int = 42):
        # Use only normal (y=0) samples as the public proxy
        if y is not None:
            normal_mask = (y == 0)
            X_pub = X[normal_mask]
        else:
            X_pub = X

        self.X   = torch.from_numpy(X_pub).float()
        self.N   = len(self.X)
        self.win = window
        self.rng = np.random.default_rng(random_state)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # Positive: a sample within ±window of anchor (temporal neighbour)
        lo  = max(0, idx - self.win)
        hi  = min(self.N - 1, idx + self.win)
        pos = int(self.rng.integers(lo, hi + 1))
        while pos == idx and lo < hi:
            pos = int(self.rng.integers(lo, hi + 1))

        # Negative: rejection sampling (far > 3×window).
        # Rejection rate ≈ 6*win/N ≪ 1 for typical N, so O(1) expected retries.
        exclusion = 3 * self.win
        if self.N <= 2 * exclusion + 1:
            # Fallback for very small datasets: just pick a different sample
            neg = (idx + self.N // 2) % self.N
        else:
            for _ in range(20):   # at most 20 retries
                neg = int(self.rng.integers(0, self.N))
                if abs(neg - idx) > exclusion:
                    break

        return self.X[idx], self.X[pos], self.X[neg]


def train_siamese(
    encoder: SiameseEncoder,
    X: np.ndarray,
    y: np.ndarray = None,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    margin: float = 0.5,
    device: str = "cpu",
    window: int = 5,
) -> SiameseEncoder:
    """
    Train the Siamese encoder using only the normal-operation (public) subset.

    Privacy guarantee: only y=0 samples are used. No client private data is
    exposed to the server during encoder pre-training. The encoder is treated
    as a public shared utility (like a pre-trained embedding model).
    Temporal proximity (consecutive windows) acts as the similarity signal,
    reflecting the natural temporal structure of IIoT sensor streams.
    """
    encoder = encoder.to(device)
    dataset = TripletDataset(X, y, window=window)
    loader  = DataLoader(dataset, batch_size=min(batch_size, len(dataset) // 2),
                         shuffle=True, drop_last=True)
    opt     = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    import logging
    log = logging.getLogger(__name__)

    n_pub = len(dataset)
    log.info("Siamese pre-training on %d public normal samples (y=0 only)", n_pub)

    encoder.train()
    for epoch in range(epochs):
        total = 0.0
        for anchor, pos, neg in loader:
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            opt.zero_grad()
            loss = loss_fn(encoder(anchor), encoder(pos), encoder(neg))
            loss.backward()
            opt.step()
            total += loss.item()
        if (epoch + 1) % 10 == 0:
            log.info("Siamese epoch %d/%d  loss=%.4f", epoch + 1, epochs,
                     total / max(1, len(loader)))

    encoder.eval()
    return encoder
