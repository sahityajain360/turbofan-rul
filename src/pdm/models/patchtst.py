"""
PatchTST regressor (Nie et al., 2023) adapted for RUL regression.

The doc's headline architecture for N-CMAPSS. Two ideas: **patching** (split each channel's
time series into sub-windows, so the Transformer attends over ~T/patch tokens instead of T —
cheaper and less noisy) and **channel independence** (the same patch-Transformer is applied to
every sensor channel, sharing weights). For a scalar RUL target we pool the per-patch
representations and combine across channels with a small head.

Input/output match the RNN/TCN regressors (``[B, T, F] -> [B]``) so it is a drop-in swap in
the sequence pipeline. The head is deliberately tiny (channel-shared projection then a linear
combine over channels) to resist overfitting on N-CMAPSS's small labeled-flight count.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PatchTSTRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        patch_len: int = 4,
        stride: int = 4,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        if seq_len < patch_len:
            raise ValueError(f"seq_len {seq_len} < patch_len {patch_len}")
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (seq_len - patch_len) // stride + 1

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, 1, self.n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.dropout = nn.Dropout(dropout)
        # channel-shared projection (per channel -> scalar), then combine across channels
        self.proj = nn.Linear(self.n_patches * d_model, 1)
        self.combine = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, F]
        B, _, F = x.shape
        x = x.transpose(1, 2)  # [B, F, T]
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)  # [B, F, N, P]
        z = self.patch_embed(patches) + self.pos  # [B, F, N, D]
        N, D = z.shape[2], z.shape[3]
        z = self.encoder(z.reshape(B * F, N, D)).reshape(B, F, N * D)  # [B, F, N*D]
        z = self.dropout(z)
        z = self.proj(z).squeeze(-1)  # [B, F]
        return self.combine(z).squeeze(-1)  # [B]
