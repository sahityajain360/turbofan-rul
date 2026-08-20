"""
Prediction-interval calibration for uncertainty-aware RUL.

MC-dropout gives a mean and a spread per engine, but the raw spread is usually
*under-confident* (intervals too narrow). We fit a single scale factor on the
spread so the nominal coverage (e.g. 80%) matches the empirical coverage on a
held-out set, then report calibrated intervals.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def gaussian_interval(mean, std, level: float = 0.8):
    """Symmetric Gaussian interval at the given coverage level. Returns (lower, upper)."""
    z = float(norm.ppf(0.5 + level / 2.0))
    mean = np.asarray(mean, float)
    std = np.asarray(std, float)
    return mean - z * std, mean + z * std


def coverage(true, lower, upper) -> float:
    """Fraction of true values inside [lower, upper]."""
    true = np.asarray(true, float)
    return float(np.mean((true >= np.asarray(lower)) & (true <= np.asarray(upper))))


def calibrate_scale(true, mean, std, level: float = 0.8, grid=None) -> float:
    """Find the std-scaling factor whose interval coverage best matches ``level``."""
    if grid is None:
        grid = np.linspace(0.3, 6.0, 115)
    true = np.asarray(true, float)
    best_s, best_gap = 1.0, float("inf")
    for s in grid:
        lo, hi = gaussian_interval(mean, np.asarray(std) * s, level)
        gap = abs(coverage(true, lo, hi) - level)
        if gap < best_gap:
            best_gap, best_s = gap, float(s)
    return best_s


def coverage_curve(true, mean, std, levels=None):
    """Empirical coverage across a range of nominal levels (for calibration plots)."""
    if levels is None:
        levels = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    emp = []
    for lvl in levels:
        lo, hi = gaussian_interval(mean, std, float(lvl))
        emp.append(coverage(true, lo, hi))
    return np.asarray(levels, float), np.asarray(emp, float)
