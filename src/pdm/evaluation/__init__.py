"""Evaluation: RUL metrics, cost-aware decisions, interval calibration, drift,
SHAP explainability, and per-engine maintenance reasoning."""
from pdm.evaluation.decision import CostModel, best_threshold, evaluate_decision
from pdm.evaluation.drift import DriftMonitor, MahalanobisOOD
from pdm.evaluation.explain import (
    EngineDrivers,
    RULExplainer,
    aggregate_to_sensors,
    global_sensor_importance,
    sensor_of,
    tree_shap,
)
from pdm.evaluation.metrics import mae, nasa_score, regression_report, rmse
from pdm.evaluation.reasoning import (
    ACTIONS,
    drift_baseline_threshold,
    maintenance_verdict,
    per_engine_psi,
    verdict_text,
)
from pdm.evaluation.uncertainty import (
    calibrate_scale,
    coverage,
    coverage_curve,
    gaussian_interval,
)

__all__ = [
    "ACTIONS",
    "CostModel",
    "DriftMonitor",
    "EngineDrivers",
    "MahalanobisOOD",
    "RULExplainer",
    "aggregate_to_sensors",
    "best_threshold",
    "calibrate_scale",
    "coverage",
    "coverage_curve",
    "drift_baseline_threshold",
    "evaluate_decision",
    "gaussian_interval",
    "global_sensor_importance",
    "mae",
    "maintenance_verdict",
    "nasa_score",
    "per_engine_psi",
    "regression_report",
    "rmse",
    "sensor_of",
    "tree_shap",
    "verdict_text",
]
