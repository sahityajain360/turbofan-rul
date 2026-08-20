"""Tests for operating-regime normalization (leak-safe, removes regime offsets)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pdm.features import RegimeNormalizer


def _two_regime_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    # regime A: op_setting_1 ~ 0, sensor ~ 100
    for i in range(60):
        rows.append({"unit": 1, "cycle": i + 1, "op_setting_1": 0.0,
                     "op_setting_2": 0.0, "op_setting_3": 0.0,
                     "sensor_1": 100.0 + rng.normal(0, 1)})
    # regime B: op_setting_1 ~ 10, sensor ~ 500 (huge offset from A)
    for i in range(60):
        rows.append({"unit": 1, "cycle": i + 61, "op_setting_1": 10.0,
                     "op_setting_2": 1.0, "op_setting_3": 0.0,
                     "sensor_1": 500.0 + rng.normal(0, 1)})
    return pd.DataFrame(rows)


def test_regime_normalization_removes_offset():
    df = _two_regime_df()
    norm = RegimeNormalizer(n_regimes=2, sensors=["sensor_1"], seed=0).fit(df)
    out = norm.transform(df)
    labels = norm.assign(df)
    for r in (0, 1):
        seg = out.loc[labels == r, "sensor_1"]
        assert abs(seg.mean()) < 0.2          # within-regime mean ~ 0
        assert abs(seg.std() - 1.0) < 0.3     # within-regime std ~ 1
    # the raw ~400-unit gap between regimes is gone
    assert out["sensor_1"].std() < 2.0


def test_single_regime_is_global_standardization():
    df = _two_regime_df()
    norm = RegimeNormalizer(n_regimes=1, sensors=["sensor_1"]).fit(df)
    out = norm.transform(df)
    np.testing.assert_allclose(out["sensor_1"].mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(out["sensor_1"].std(), 1.0, atol=1e-6)
