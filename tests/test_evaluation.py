"""Tests for RUL metrics and the cost-aware decision layer."""
from __future__ import annotations

import numpy as np

from pdm.evaluation import (
    CostModel,
    best_threshold,
    evaluate_decision,
    nasa_score,
    rmse,
)


def test_nasa_score_perfect_is_zero():
    assert nasa_score([50, 100, 20], [50, 100, 20]) == 0.0


def test_nasa_score_penalizes_overestimation_more():
    # predicting 10 too HIGH (risky) must cost more than 10 too LOW (safe)
    over = nasa_score([50], [60])
    under = nasa_score([50], [40])
    assert over > under
    np.testing.assert_allclose(over, np.exp(10 / 10) - 1)
    np.testing.assert_allclose(under, np.exp(10 / 13) - 1)


def test_rmse_basic():
    np.testing.assert_allclose(rmse([0, 0, 0], [3, 0, 4]), np.sqrt((9 + 16) / 3))


def test_evaluate_decision_confusion_and_cost():
    # truth RUL, pred RUL; horizon = threshold = 30
    true_rul = [10, 40, 25, 90]   # needs maintenance: [T, F, T, F]
    pred_rul = [12, 20, 50, 80]   # alert (<=30):       [T, T, F, F]
    r = evaluate_decision(true_rul, pred_rul, horizon=30, threshold=30,
                          cost=CostModel(c_unplanned_failure=100, c_early_maintenance=10))
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (1, 1, 1, 1)
    # cost = 1 missed failure * 100 + 1 false alarm * 10
    assert r["total_cost"] == 110.0


def test_best_threshold_minimizes_cost():
    rng = np.random.default_rng(0)
    true_rul = rng.uniform(0, 120, size=500)
    pred_rul = true_rul + rng.normal(0, 8, size=500)  # decent predictor
    best = best_threshold(true_rul, pred_rul, horizon=30,
                          cost=CostModel(), thresholds=np.arange(5, 80, 1))
    # a fixed mid threshold should not beat the optimized one
    fixed = evaluate_decision(true_rul, pred_rul, horizon=30, threshold=30)
    assert best["total_cost"] <= fixed["total_cost"]
