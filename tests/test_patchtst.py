"""Shape/smoke tests for the PatchTST regressor (CPU, no data needed)."""
from __future__ import annotations

import pytest
import torch

from pdm.models.patchtst import PatchTSTRegressor


def test_forward_shape_and_patch_count():
    model = PatchTSTRegressor(n_features=8, seq_len=20, patch_len=4, stride=4)
    assert model.n_patches == 5  # (20-4)//4 + 1
    out = model(torch.randn(6, 20, 8))
    assert out.shape == (6,)
    assert torch.isfinite(out).all()


def test_backward_runs():
    model = PatchTSTRegressor(n_features=4, seq_len=16, patch_len=4, stride=2)
    out = model(torch.randn(3, 16, 4))
    out.sum().backward()  # gradients flow
    assert any(p.grad is not None for p in model.parameters())


def test_rejects_seq_shorter_than_patch():
    with pytest.raises(ValueError):
        PatchTSTRegressor(n_features=4, seq_len=3, patch_len=4)
