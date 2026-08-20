"""
Sliding-window sequences for deep RUL models.

Deep models consume the raw (regime-normalized) sensor sequence directly rather than
hand-built summary statistics. We turn each engine's multivariate time series into
fixed-length windows:

  * **Training windows** — every length-``window`` slice of an engine, labeled with the
    RUL at the slice's *last* cycle. Strictly trailing, never crossing an engine.
  * **Last window per engine** — the final ``window`` cycles of each engine (the
    standard C-MAPSS test snapshot), left-padded by repeating the earliest cycle when
    an engine is shorter than ``window``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_training_windows(
    df: pd.DataFrame, sensors: list[str], window: int, stride: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X, y, units)``: X is (N, window, n_sensors), y is RUL at window end."""
    Xs, ys, us = [], [], []
    for unit, g in df.groupby("unit", sort=False):
        g = g.sort_values("cycle")
        vals = g[sensors].to_numpy(dtype=np.float32)
        ruls = g["rul"].to_numpy(dtype=np.float32)
        n = len(g)
        for end in range(window, n + 1, stride):
            Xs.append(vals[end - window : end])
            ys.append(ruls[end - 1])
            us.append(int(unit))
    if not Xs:
        return (
            np.empty((0, window, len(sensors)), np.float32),
            np.empty((0,), np.float32),
            np.empty((0,), int),
        )
    return np.stack(Xs), np.asarray(ys, np.float32), np.asarray(us, int)


def make_last_windows(
    df: pd.DataFrame, sensors: list[str], window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Final ``window`` cycles per engine (left-padded). Returns ``(X, units)`` in
    ascending unit order — aligned with ``load_test_rul``."""
    Xs, us = [], []
    for unit in sorted(df["unit"].unique()):
        g = df[df["unit"] == unit].sort_values("cycle")
        vals = g[sensors].to_numpy(dtype=np.float32)
        n = len(vals)
        if n >= window:
            win = vals[-window:]
        else:  # left-pad by repeating the earliest available cycle
            pad = np.repeat(vals[:1], window - n, axis=0)
            win = np.concatenate([pad, vals], axis=0)
        Xs.append(win)
        us.append(int(unit))
    return np.stack(Xs), np.asarray(us, int)
