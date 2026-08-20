"""Shape and smoke tests for Mamba sequence models (runs on GPU if available)."""
from __future__ import annotations

import pytest

pytest.importorskip("mamba_ssm", reason="mamba-ssm only installed in the WSL CUDA env")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from pdm.models.mamba import MambaTelemetryEncoder, NextStepForecastingModel  # noqa: E402
from pdm.models.ssl import MaskedSensorModel, masked_mse, random_mask  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture(autouse=True)
def skip_if_no_cuda():
    if DEVICE == "cpu":
        pytest.skip("Mamba CUDA kernels require a GPU and cannot run on CPU.")


def test_mamba_encoder_and_heads_shapes():
    enc = MambaTelemetryEncoder(n_features=6, d_model=32, n_layers=2, bidirectional=False).to(DEVICE)
    x = torch.randn(5, 40, 6, device=DEVICE)
    assert enc(x).shape == (5, 40, 32)
    assert MaskedSensorModel(enc, 6).to(DEVICE)(x).shape == (5, 40, 6)  # reconstruction
    
    # Next step forecasting model shape
    forecaster = NextStepForecastingModel(enc, 6).to(DEVICE)
    assert forecaster(x).shape == (5, 39, 6)  # predicts 39 steps for input of 40 steps


def test_bidirectional_mamba_shapes():
    enc = MambaTelemetryEncoder(n_features=8, d_model=16, n_layers=1, bidirectional=True).to(DEVICE)
    x = torch.randn(4, 30, 8, device=DEVICE)
    assert enc(x).shape == (4, 30, 16)


def test_mamba_pretrain_backward():
    enc = MambaTelemetryEncoder(n_features=4, d_model=16, n_layers=1).to(DEVICE)
    model = MaskedSensorModel(enc, 4).to(DEVICE)
    x = torch.randn(8, 20, 4, device=DEVICE)
    xm, mask = random_mask(x, ratio=0.3)
    loss = masked_mse(model(xm), x, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.parameters())


def test_mamba_forecasting_backward():
    enc = MambaTelemetryEncoder(n_features=4, d_model=16, n_layers=1).to(DEVICE)
    model = NextStepForecastingModel(enc, 4).to(DEVICE)
    x = torch.randn(8, 20, 4, device=DEVICE)
    pred = model(x)
    target = x[:, 1:]
    loss = nn.functional.mse_loss(pred, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.parameters())
