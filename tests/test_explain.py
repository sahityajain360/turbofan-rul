"""Tests for SHAP explainability helpers (model-agnostic core)."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from pdm.evaluation.explain import (
    aggregate_to_sensors,
    global_sensor_importance,
    sensor_of,
    tree_shap,
)
from pdm.features import feature_columns


def test_sensor_of_strips_stat():
    assert sensor_of("sensor_11_slope") == "sensor_11"
    assert sensor_of("sensor_4_mean") == "sensor_4"


def test_aggregate_to_sensors_sums_four_stats():
    sensors = ["sensor_1", "sensor_2"]
    cols = feature_columns(sensors)  # 8 cols, sensor_1's four then sensor_2's four
    values = np.array([[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]])
    out = aggregate_to_sensors(values, cols, sensors)
    assert out.loc[0, "sensor_1"] == 10.0   # 1+2+3+4
    assert out.loc[0, "sensor_2"] == 100.0  # 10+20+30+40


def test_global_importance_uses_abs_and_sorts():
    sensors = ["sensor_1", "sensor_2"]
    cols = feature_columns(sensors)
    # sensor_2 has larger |contributions| -> should rank first
    values = np.array([[1.0, -1.0, 0.0, 0.0, -5.0, 5.0, 0.0, 0.0]])
    per_feature, per_sensor = global_sensor_importance(values, cols, sensors)
    assert list(per_sensor)[0] == "sensor_2"          # sorted descending
    assert per_sensor["sensor_1"] == 2.0              # |1| + |-1|
    assert per_sensor["sensor_2"] == 10.0             # |-5| + |5|
    assert per_feature["sensor_2_mean"] == 5.0


def test_tree_shap_additivity():
    """TreeExplainer guarantees base + sum(shap) == model output."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 5)), columns=[f"f{i}" for i in range(5)])
    y = 3 * X["f0"] - 2 * X["f1"] + rng.normal(scale=0.1, size=len(X))
    model = lgb.LGBMRegressor(n_estimators=50, num_leaves=8, verbose=-1).fit(X, y)

    values, base = tree_shap(model, X)
    reconstructed = base + values.sum(axis=1)
    assert np.allclose(reconstructed, model.predict(X), atol=1e-4)
