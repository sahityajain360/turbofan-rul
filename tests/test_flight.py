"""Tests for the N-CMAPSS per-flight reduction + flight-history windowing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pdm.features.flight import (
    add_flight_deltas,
    flight_summary,
    make_flight_windows,
    make_within_flight_windows,
    reduce_to_flights,
)


def test_reduce_to_flights_means_one_row_per_flight():
    df = pd.DataFrame({
        "unit": [1, 1, 1, 1, 2, 2],
        "cycle": [1, 1, 2, 2, 1, 1],
        "s1": [10.0, 20.0, 0.0, 4.0, 5.0, 5.0],
        "rul": [50, 50, 49, 49, 8, 8],
    })
    out = reduce_to_flights(df, ["s1"]).set_index(["unit", "cycle"])
    assert len(out) == 3  # (1,1), (1,2), (2,1)
    assert out.loc[(1, 1), "s1"] == 15.0  # mean(10, 20)
    assert out.loc[(1, 2), "s1"] == 2.0   # mean(0, 4)
    assert out.loc[(2, 1), "s1"] == 5.0
    assert out.loc[(1, 1), "rul"] == 50   # rul constant within a flight


def test_make_flight_windows_shapes_padding_alignment():
    flights = pd.DataFrame({
        "unit": [1, 1, 1, 2, 2],
        "cycle": [1, 2, 3, 1, 2],
        "s1": [1.0, 2.0, 3.0, 7.0, 8.0],
        "s2": [10.0, 20.0, 30.0, 70.0, 80.0],
        "rul": [2, 1, 0, 1, 0],
    })
    X, y, u, c = make_flight_windows(flights, ["s1", "s2"], window=2)
    assert X.shape == (5, 2, 2)              # one window per flight, len-2, 2 sensors
    assert u.tolist() == [1, 1, 1, 2, 2]
    assert c.tolist() == [1, 2, 3, 1, 2]
    assert y.tolist() == [2.0, 1.0, 0.0, 1.0, 0.0]  # label = current flight's RUL
    # first flight left-padded by repeating itself
    np.testing.assert_array_equal(X[0], [[1, 10], [1, 10]])
    # trailing window includes the current flight as the last step
    np.testing.assert_array_equal(X[1], [[1, 10], [2, 20]])
    np.testing.assert_array_equal(X[2], [[2, 20], [3, 30]])
    # unit 2 window never borrows unit 1 flights (no cross-engine leakage)
    np.testing.assert_array_equal(X[3], [[7, 70], [7, 70]])


def test_flight_summary_stats_and_channels():
    df = pd.DataFrame({
        "unit": [1, 1, 1, 1],
        "cycle": [1, 1, 2, 2],
        "s1": [10.0, 20.0, 0.0, 4.0],
        "rul": [5, 5, 4, 4],
    })
    out, channels = flight_summary(df, ["s1"], stats=("mean", "std", "min", "max"))
    assert channels == ["s1_mean", "s1_std", "s1_min", "s1_max"]
    r = out.set_index("cycle")
    assert r.loc[1, "s1_mean"] == 15.0 and r.loc[1, "s1_min"] == 10.0 and r.loc[1, "s1_max"] == 20.0
    assert r.loc[2, "s1_mean"] == 2.0
    assert r.loc[1, "rul"] == 5


def test_add_flight_deltas_is_trailing_per_unit():
    flights = pd.DataFrame({
        "unit": [1, 1, 1, 2, 2],
        "cycle": [1, 2, 3, 1, 2],
        "m": [10.0, 13.0, 18.0, 100.0, 90.0],
    })
    out, new = add_flight_deltas(flights, ["m"], lag=1)
    assert new == ["m_d1"]
    # first flight per unit -> 0; subsequent -> difference from previous flight
    assert out["m_d1"].tolist() == [0.0, 3.0, 5.0, 0.0, -10.0]


def test_make_within_flight_windows_slices_and_inherits_rul():
    # one flight (unit 1, cycle 1) with 10 in-order samples; one short flight skipped
    rows = []
    for t in range(10):
        rows.append({"unit": 1, "cycle": 1, "s1": float(t), "rul": 7})
    for t in range(3):
        rows.append({"unit": 1, "cycle": 2, "s1": 100.0 + t, "rul": 6})  # too short -> skipped
    df = pd.DataFrame(rows)
    X, y, u, c = make_within_flight_windows(df, ["s1"], length=4, stride=3)
    # starts 0,3,6 (7-? : range(0, 10-4+1=7, 3) -> 0,3,6) => 3 windows, all from flight (1,1)
    assert X.shape == (3, 4, 1)
    assert (y == 7).all() and (u == 1).all() and (c == 1).all()
    np.testing.assert_array_equal(X[0].ravel(), [0, 1, 2, 3])   # in flight-time order
    np.testing.assert_array_equal(X[1].ravel(), [3, 4, 5, 6])


def test_make_flight_windows_window_longer_than_engine():
    flights = pd.DataFrame({
        "unit": [1, 1], "cycle": [1, 2], "s1": [3.0, 5.0], "rul": [1, 0],
    })
    X, y, u, c = make_flight_windows(flights, ["s1"], window=4)
    assert X.shape == (2, 4, 1)
    # second flight, window 4: pad earliest (flight1) x2 then flight1, flight2
    np.testing.assert_array_equal(X[1].ravel(), [3, 3, 3, 5])
