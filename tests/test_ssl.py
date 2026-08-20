"""Shape/smoke tests for the self-supervised telemetry modules (CPU, no data)."""
from __future__ import annotations

import torch

from pdm.models.ssl import (
    MaskedSensorModel,
    RULRegressor,
    TelemetryEncoder,
    masked_mse,
    random_mask,
)


def test_encoder_and_heads_shapes():
    enc = TelemetryEncoder(n_features=6, d_model=32, n_heads=4, n_layers=2)
    x = torch.randn(5, 40, 6)
    assert enc(x).shape == (5, 40, 32)
    assert MaskedSensorModel(enc, 6)(x).shape == (5, 40, 6)  # reconstruction
    assert RULRegressor(enc)(x).shape == (5,)               # scalar RUL per window


def test_pretrain_step_backward():
    enc = TelemetryEncoder(n_features=4, d_model=16, n_layers=1)
    model = MaskedSensorModel(enc, 4)
    x = torch.randn(8, 20, 4)
    xm, mask = random_mask(x, ratio=0.3)
    loss = masked_mse(model(xm), x, mask)
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None for p in model.parameters())


def test_random_mask_ratio_and_fill():
    torch.manual_seed(0)
    x = torch.randn(2, 50, 8)
    xm, mask = random_mask(x, ratio=0.25)
    assert abs(mask.float().mean().item() - 0.25) < 0.05   # ~25% masked
    assert (xm[mask] == 0).all()                            # masked entries zeroed
    assert torch.equal(xm[~mask], x[~mask])                 # rest untouched


def test_masked_mse_only_counts_masked():
    pred = torch.zeros(1, 3, 2)
    target = torch.ones(1, 3, 2)
    mask = torch.zeros(1, 3, 2, dtype=torch.bool)
    mask[0, 0, 0] = True  # a single masked entry with error 1
    assert torch.isclose(masked_mse(pred, target, mask), torch.tensor(1.0))
