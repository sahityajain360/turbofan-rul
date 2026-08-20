"""Tests for the PSI drift monitor + Mahalanobis OOD gate."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pdm.evaluation.drift import DriftMonitor, MahalanobisOOD


def test_no_drift_on_same_distribution():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 5000), "b": rng.normal(5, 2, 5000)})
    new = pd.DataFrame({"a": rng.normal(0, 1, 5000), "b": rng.normal(5, 2, 5000)})
    mon = DriftMonitor(["a", "b"]).fit(ref)
    s = mon.score(new)
    assert s["psi_max"] < 0.1
    assert not s["flag"]


def test_detects_shift():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 5000), "b": rng.normal(0, 1, 5000)})
    new = pd.DataFrame({"a": rng.normal(3, 1, 5000), "b": rng.normal(0, 1, 5000)})  # 'a' shifted
    mon = DriftMonitor(["a", "b"]).fit(ref)
    s = mon.score(new)
    assert s["flag"]
    assert "a" in s["drifted_features"]
    assert s["psi_per_feature"]["a"] > s["psi_per_feature"]["b"]


def test_mahalanobis_catches_correlation_shift_psi_misses():
    """Same marginals, broken correlation: Mahalanobis flags it; per-feature PSI doesn't."""
    rng = np.random.default_rng(3)
    n = 8000
    a = rng.normal(0, 1, n)
    ref = pd.DataFrame({"a": a, "b": a + rng.normal(0, 0.2, n)})  # a,b strongly correlated
    a2 = rng.normal(0, 1, n)
    # OOD: identical per-feature marginals (~N(0,1) each) but the correlation is inverted
    ood = pd.DataFrame({"a": a2, "b": -a2 + rng.normal(0, 0.2, n)})

    maha = MahalanobisOOD(["a", "b"]).fit(ref)
    d_in = maha.score(ref)["maha_mean"]
    d_ood = maha.score(ood)["maha_mean"]
    assert d_ood > 3 * d_in  # correlation break is far out-of-distribution

    # PSI (marginals) barely notices — the point of using Mahalanobis
    psi = DriftMonitor(["a", "b"]).fit(ref).score(ood)
    assert psi["psi_max"] < 0.25
