"""RUL regression metrics, including the asymmetric NASA/PHM08 score."""
from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    e = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sqrt(np.mean(e**2)))


def mae(y_true, y_pred) -> float:
    e = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.mean(np.abs(e)))


def nasa_score(y_true, y_pred) -> float:
    """NASA/PHM08 prognostics score (lower is better; 0 = perfect).

    With ``d = pred - true``, **over-estimating** RUL (d > 0, i.e. predicting more
    remaining life than there is — the dangerous direction) is penalized harder
    than under-estimating:

        d < 0 :  exp(-d / 13) - 1     (early / conservative)
        d >= 0:  exp( d / 10) - 1     (late / risky)
    """
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(s))


def regression_report(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
        "n": int(len(np.asarray(y_true))),
    }
