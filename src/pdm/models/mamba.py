"""
Mamba-based State Space Model (SSM) sequence encoder for predictive maintenance.
Provides a drop-in replacement for TelemetryEncoder in SSL pretraining.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    """Mamba block with optional bidirectionality."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.fwd_mamba = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        if bidirectional:
            self.bwd_mamba = Mamba(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
            )
            self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        out_fwd = self.fwd_mamba(x)
        if self.bidirectional:
            x_flipped = x.flip(dims=[1])
            out_bwd = self.bwd_mamba(x_flipped).flip(dims=[1])
            return self.proj(torch.cat([out_fwd, out_bwd], dim=-1))
        else:
            return out_fwd


class MambaTelemetryEncoder(nn.Module):
    """Mamba sequence encoder that maps [B, L, F] -> [B, L, d_model]."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.input = nn.Linear(n_features, d_model)

        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                bidirectional=bidirectional,
            )
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, L, F]
        h = self.input(x)  # [B, L, d_model]
        for layer in self.layers:
            h = h + self.dropout(layer(h))  # residual connection + dropout
        return self.norm(h)


class NextStepForecastingModel(nn.Module):
    """Encoder + next-step forecasting head.
    Given [B, L, F], passes it to the encoder to get [B, L, d_model],
    then projects it to predict the next step [B, L-1, F].
    """

    def __init__(self, encoder: nn.Module, n_features: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.d_model, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F]
        h = self.encoder(x)  # [B, L, d_model]
        # Predict t=1..L from hidden states of t=0..L-1
        pred = self.head(h[:, :-1])  # [B, L-1, F]
        return pred

