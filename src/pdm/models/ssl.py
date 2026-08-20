"""
Self-supervised pretraining for within-flight telemetry (M7.5) — "BERT for sensors".

Motivation (from M7.2/M7.3): RUL labels are scarce (~300 flight-cycles) but the *unlabeled*
within-flight 1 Hz stream is abundant. We pretrain a Transformer encoder with **masked sensor
modeling** — randomly zero out sensor readings in a within-flight window and reconstruct them —
then fine-tune the encoder with an RUL head. The clean test is pretrained-then-fine-tuned vs the
identical model trained from scratch.

Pieces:
  * ``TelemetryEncoder`` — Transformer encoder over a window ``[B, L, F] -> [B, L, d_model]``.
  * ``MaskedSensorModel`` — encoder + linear head reconstructing masked sensor values.
  * ``RULRegressor`` — encoder + mean-pool + MLP head -> scalar RUL.
  * ``random_mask`` / ``masked_mse`` — entry-wise masking and reconstruction loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TelemetryEncoder(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        self.d_model = d_model
        self.input = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, L, F] -> [B, L, d_model]
        L = x.shape[1]
        return self.encoder(self.input(x) + self.pos[:, :L])


class MaskedSensorModel(nn.Module):
    """Encoder + linear reconstruction head (input is the already-masked window)."""

    def __init__(self, encoder: TelemetryEncoder, n_features: int):
        super().__init__()
        self.encoder = encoder
        self.recon = nn.Linear(encoder.d_model, n_features)

    def forward(self, x_masked: torch.Tensor) -> torch.Tensor:  # [B, L, F] -> [B, L, F]
        return self.recon(self.encoder(x_masked))


class RULRegressor(nn.Module):
    """Encoder + mean-pool over time + MLP head -> scalar RUL ([B, L, F] -> [B])."""

    def __init__(self, encoder: TelemetryEncoder, dropout: float = 0.2):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x).mean(dim=1)).squeeze(-1)


def random_mask(x: torch.Tensor, ratio: float = 0.25):
    """Entry-wise mask: zero a random ``ratio`` of (timestep, sensor) entries.

    Returns ``(x_masked, mask)`` where ``mask`` is True at the masked positions (those the
    reconstruction loss is computed on).
    """
    mask = torch.rand_like(x) < ratio
    return x.masked_fill(mask, 0.0), mask


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE over masked entries only."""
    se = (pred - target) ** 2 * mask
    return se.sum() / mask.sum().clamp_min(1)
