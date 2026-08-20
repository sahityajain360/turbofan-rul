"""Tests for sliding-window sequence construction (shapes, labels, padding, no leak)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pdm.data.sequences import make_last_windows, make_training_windows


def _df(units_lengths: dict[int, int]) -> pd.DataFrame:
    rows = []
    for u, n in units_lengths.items():
        for c in range(1, n + 1):
            rows.append({"unit": u, "cycle": c, "s": float(u * 100 + c), "rul": n - c})
    return pd.DataFrame(rows)


def test_training_window_shapes_and_labels():
    df = _df({1: 5})
    X, y, units = make_training_windows(df, ["s"], window=3)
    # windows end at cycles 3,4,5 -> 3 windows of shape (3,1)
    assert X.shape == (3, 3, 1)
    # label is RUL at the window's last cycle: cycles 3,4,5 -> rul 2,1,0
    np.testing.assert_array_equal(y, [2.0, 1.0, 0.0])
    assert set(units) == {1}


def test_no_cross_unit_windows():
    df = _df({1: 4, 2: 4})
    X, y, units = make_training_windows(df, ["s"], window=3)
    # each unit yields windows for ends 3,4 -> 2 each, 4 total; none mixes units
    assert X.shape[0] == 4
    for win in X:
        col = win[:, 0]
        assert (col >= 100).all() and (col < 200).all() or (col >= 200).all()


def test_last_window_left_pads_short_engine():
    df = _df({1: 2})  # shorter than window
    X, units = make_last_windows(df, ["s"], window=4)
    assert X.shape == (1, 4, 1)
    # first two rows are the repeated earliest cycle (value 101), then 101,102
    np.testing.assert_array_equal(X[0, :, 0], [101.0, 101.0, 101.0, 102.0])
    assert list(units) == [1]


def test_last_window_units_sorted():
    df = _df({3: 6, 1: 6})
    X, units = make_last_windows(df, ["s"], window=3)
    assert list(units) == [1, 3]  # ascending, aligns with RUL truth file
