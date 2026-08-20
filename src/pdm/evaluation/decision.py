"""
Maintenance-decision layer + cost model.

This is what turns a RUL number into an *action*. An engine "needs maintenance"
if its true RUL is within ``horizon`` cycles of failure; the model raises an alert
when predicted RUL <= ``threshold``. Because a missed failure is far more expensive
than an early replacement, we evaluate decisions with an **asymmetric cost model**
and tune the alert threshold to minimize expected cost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostModel:
    """Relative costs of each decision outcome (unitless, tune to the asset)."""

    c_unplanned_failure: float = 100.0  # missed warning -> ran to failure (worst)
    c_early_maintenance: float = 10.0   # false alarm -> serviced too early


def evaluate_decision(
    true_rul,
    pred_rul,
    horizon: float = 30.0,
    threshold: float | None = None,
    cost: CostModel | None = None,
) -> dict[str, float]:
    """Confusion-matrix + cost view of the "maintain now?" decision.

    ``threshold`` defaults to ``horizon`` (alert exactly when predicted to be
    within the maintenance horizon).
    """
    cost = cost or CostModel()
    threshold = horizon if threshold is None else threshold
    true_rul = np.asarray(true_rul, float)
    pred_rul = np.asarray(pred_rul, float)

    needs = true_rul <= horizon   # ground-truth: within maintenance horizon
    alert = pred_rul <= threshold  # model decision

    tp = int(np.sum(alert & needs))
    fp = int(np.sum(alert & ~needs))
    fn = int(np.sum(~alert & needs))
    tn = int(np.sum(~alert & ~needs))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    total_cost = fn * cost.c_unplanned_failure + fp * cost.c_early_maintenance

    return {
        "horizon": float(horizon),
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "total_cost": float(total_cost),
    }


def best_threshold(
    true_rul,
    pred_rul,
    horizon: float = 30.0,
    cost: CostModel | None = None,
    thresholds=None,
) -> dict[str, float]:
    """Sweep alert thresholds and return the decision dict with minimum cost.

    Ties broken toward the *higher* threshold (warn earlier — the safe side).
    """
    cost = cost or CostModel()
    if thresholds is None:
        thresholds = np.arange(5.0, 80.0, 1.0)
    best = None
    for t in thresholds:
        r = evaluate_decision(true_rul, pred_rul, horizon, float(t), cost)
        if best is None or r["total_cost"] <= best["total_cost"]:
            best = r
    return best
