"""Anti-leakage + correctness tests for rolling-window features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pdm.features import add_rolling_features, feature_columns, last_per_unit


def _toy() -> pd.DataFrame:
    # unit 1: huge values, unit 2: small values — used to prove no cross-unit leak
    rows = []
    for c in range(1, 6):
        rows.append({"unit": 1, "cycle": c, "sensor_x": 1000.0 + c, "rul": 5 - c})
    for c in range(1, 6):
        rows.append({"unit": 2, "cycle": c, "sensor_x": c, "rul": 5 - c})
    return pd.DataFrame(rows)


def test_first_cycle_uses_only_present_value():
    feats = add_rolling_features(_toy(), ["sensor_x"], window=3)
    first = feats[feats["cycle"] == 1].sort_values("unit")
    # at the very first cycle: mean == last == value, std == 0, slope == 0
    np.testing.assert_allclose(first["sensor_x_mean"], first["sensor_x_last"])
    np.testing.assert_allclose(first["sensor_x_std"], 0.0)
    np.testing.assert_allclose(first["sensor_x_slope"], 0.0)


def test_no_cross_unit_leakage():
    feats = add_rolling_features(_toy(), ["sensor_x"], window=10)
    u2 = feats[feats["unit"] == 2]
    # unit 2's rolling mean must never be polluted by unit 1's huge values
    assert u2["sensor_x_mean"].max() < 100.0


def test_trailing_window_and_slope():
    feats = add_rolling_features(_toy(), ["sensor_x"], window=3)
    u2 = feats[feats["unit"] == 2].sort_values("cycle").reset_index(drop=True)
    # window=3 trailing mean at cycle 5 over values {3,4,5} == 4
    assert u2.loc[u2["cycle"] == 5, "sensor_x_mean"].iloc[0] == 4.0
    # sensor_x increases by 1 each cycle -> slope ~ 1
    np.testing.assert_allclose(u2.loc[u2["cycle"] == 5, "sensor_x_slope"].iloc[0], 1.0, atol=1e-6)


def test_feature_columns_and_last_per_unit():
    sensors = ["sensor_a", "sensor_b"]
    assert feature_columns(sensors) == [
        "sensor_a_mean", "sensor_a_std", "sensor_a_last", "sensor_a_slope",
        "sensor_b_mean", "sensor_b_std", "sensor_b_last", "sensor_b_slope",
    ]
    last = last_per_unit(_toy())
    assert len(last) == 2
    assert set(last["cycle"]) == {5}
