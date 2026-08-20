"""Sanity + anti-leakage tests for the C-MAPSS data layer.

These encode the project's guardrails as executable checks:
  * correct schema parsing,
  * piecewise-linear RUL that is capped and non-increasing within an engine,
  * engine-wise splits that are disjoint by unit.
"""
from __future__ import annotations

import numpy as np
import pytest

from pdm.data import cmapss


@pytest.fixture(scope="module")
def fd001_train():
    return cmapss.load_raw("FD001", "train")


def test_schema(fd001_train):
    df = fd001_train
    assert list(df.columns) == cmapss.ALL_COLUMNS
    assert df.shape[1] == 26
    assert not df.isna().any().any()
    assert df["unit"].dtype.kind == "i"
    assert df["cycle"].dtype.kind == "i"


def test_train_rul_capped_and_monotonic(fd001_train):
    cap = 125
    df = cmapss.add_train_rul(fd001_train, cap=cap)
    assert df["rul"].max() == cap
    # last cycle of every engine must be RUL == 0 (failure point)
    last = df.sort_values("cycle").groupby("unit").tail(1)
    assert (last["rul"] == 0).all()
    # within each engine RUL is non-increasing over cycles
    for _, g in df.sort_values("cycle").groupby("unit"):
        assert (g["rul"].diff().dropna() <= 0).all()


def test_test_rul_matches_endpoint():
    df = cmapss.load_raw("FD001", "test")
    rul_end = cmapss.load_test_rul("FD001")
    assert len(rul_end) == df["unit"].nunique()
    labeled = cmapss.add_test_rul(df, rul_end, cap=None)
    # at each unit's last cycle, RUL must equal the provided endpoint value
    last = labeled.sort_values("cycle").groupby("unit").tail(1).sort_values("unit")
    np.testing.assert_array_equal(last["rul"].to_numpy(), rul_end)


def test_split_is_disjoint_and_covering(fd001_train):
    units = fd001_train["unit"].unique()
    train_u, val_u = cmapss.split_units(units, val_fraction=0.2, seed=42)
    assert train_u.isdisjoint(val_u)
    assert train_u | val_u == set(int(u) for u in units)
    # deterministic given the seed
    train_u2, val_u2 = cmapss.split_units(units, val_fraction=0.2, seed=42)
    assert val_u == val_u2
