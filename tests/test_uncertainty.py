"""Tests for prediction-interval calibration."""
from __future__ import annotations

import numpy as np

from pdm.evaluation.uncertainty import calibrate_scale, coverage, gaussian_interval


def test_coverage_bounds():
    assert coverage([1, 2, 3], [0, 0, 0], [5, 5, 5]) == 1.0
    assert coverage([10], [0], [5]) == 0.0


def test_gaussian_interval_width():
    lo, hi = gaussian_interval(np.array([0.0]), np.array([1.0]), level=0.8)
    # 80% Gaussian interval half-width ~ 1.2816 sigma
    assert abs((hi[0] - lo[0]) - 2 * 1.2816) < 0.02


def test_calibrate_scale_recovers_underconfidence():
    rng = np.random.default_rng(0)
    n = 5000
    mean = np.zeros(n)
    std = np.ones(n)              # reported spread
    true = rng.normal(0, 2.0, n)  # actual spread is 2x -> intervals too narrow
    s = calibrate_scale(true, mean, std, level=0.8)
    assert 1.6 < s < 2.4          # calibration should roughly recover the 2x factor
