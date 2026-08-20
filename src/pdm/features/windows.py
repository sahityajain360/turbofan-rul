"""
Leak-safe rolling-window features for tabular RUL baselines.

For each ``(unit, cycle)`` row we summarize a *trailing* window of the last
``window`` cycles **of that engine only**:

  * a feature at cycle ``t`` uses only cycles ``<= t``  (no future leakage),
  * windows never cross an engine boundary (per-unit grouping).

Stats per sensor: rolling ``mean``, ``std``, current ``last`` value, and a
trailing linear ``slope`` (degradation trend).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_STATS = ("mean", "std", "last", "slope")


def _rolling_slope(cycle: pd.Series, x: pd.Series, window: int) -> np.ndarray:
    """Trailing least-squares slope of ``x`` vs ``cycle`` over ``window`` (vectorized).

    slope = cov(cycle, x) / var(cycle) computed on the trailing window.
    """
    c = cycle.astype(float)
    x = x.astype(float)
    mean_c = c.rolling(window, min_periods=2).mean()
    mean_x = x.rolling(window, min_periods=2).mean()
    mean_cx = (c * x).rolling(window, min_periods=2).mean()
    mean_cc = (c * c).rolling(window, min_periods=2).mean()
    var_c = mean_cc - mean_c**2
    cov = mean_cx - mean_c * mean_x
    slope = cov / var_c.replace(0.0, np.nan)
    return slope.fillna(0.0).to_numpy()


def add_rolling_features(
    df: pd.DataFrame,
    sensors: list[str],
    window: int = 30,
    keep: tuple[str, ...] = ("unit", "cycle", "rul"),
) -> pd.DataFrame:
    """Return a new frame with rolling-window features for each sensor.

    ``keep`` columns (those that exist) are carried through unchanged.
    """
    keep = [c for c in keep if c in df.columns]
    blocks = []
    for _, g in df.groupby("unit", sort=False):
        g = g.sort_values("cycle")
        cyc = g["cycle"]
        feat: dict[str, np.ndarray] = {}
        for s in sensors:
            x = g[s].astype(float)
            r = x.rolling(window, min_periods=1)
            feat[f"{s}_mean"] = r.mean().to_numpy()
            feat[f"{s}_std"] = r.std().fillna(0.0).to_numpy()
            feat[f"{s}_last"] = x.to_numpy()
            feat[f"{s}_slope"] = _rolling_slope(cyc, x, window)
        block = pd.concat(
            [g[keep].reset_index(drop=True), pd.DataFrame(feat)], axis=1
        )
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def feature_columns(sensors: list[str]) -> list[str]:
    """Names of the engineered feature columns, in generation order."""
    return [f"{s}_{stat}" for s in sensors for stat in _STATS]


def last_per_unit(df: pd.DataFrame) -> pd.DataFrame:
    """Last recorded cycle per engine (the standard C-MAPSS test snapshot)."""
    return (
        df.sort_values("cycle")
        .groupby("unit", as_index=False)
        .tail(1)
        .sort_values("unit")
        .reset_index(drop=True)
    )
