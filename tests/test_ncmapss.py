"""Tests for the N-CMAPSS HDF5 data layer.

Skipped automatically when the (large, ~2.4 GB) DS02 file is not present, so the suite
stays portable; runs in full locally where the data has been downloaded.
"""
from __future__ import annotations

import numpy as np
import pytest

from pdm import paths
from pdm.data import ncmapss

pytestmark = pytest.mark.skipif(
    not paths.NCMAPSS_DS02.exists(),
    reason="N-CMAPSS DS02 HDF5 not present (large file, not committed)",
)


def test_load_columns_units_and_no_leakage():
    df = ncmapss.load_ncmapss("test", subsample=500)
    expected = set(ncmapss.A_VARS + ncmapss.W_VARS + ncmapss.XS_VARS + ["rul"])
    assert expected.issubset(df.columns)
    # T (health params) are simulator ground truth -> must never appear as features
    assert not any(v in df.columns for v in ncmapss.T_VARS)
    assert sorted(df["unit"].unique()) == [11, 14, 15]
    assert df["rul"].min() >= 0
    for c in ("unit", "cycle", "Fc", "hs"):
        assert np.issubdtype(df[c].dtype, np.integer)


def test_include_virtual_and_health_params():
    df = ncmapss.load_ncmapss(
        "test", subsample=1000, include_virtual=True, include_health_params=True
    )
    assert set(ncmapss.XV_VARS).issubset(df.columns)
    assert set(ncmapss.T_VARS).issubset(df.columns)


def test_subsample_reduces_rows():
    assert len(ncmapss.load_ncmapss("test", subsample=500)) > len(
        ncmapss.load_ncmapss("test", subsample=1000)
    )


def test_unit_ids_match_predefined_split():
    assert ncmapss.unit_ids("dev").tolist() == [2, 5, 10, 16, 18, 20]
    assert ncmapss.unit_ids("test").tolist() == [11, 14, 15]


def test_feature_sensors():
    assert ncmapss.feature_sensors() == ncmapss.W_VARS + ncmapss.XS_VARS
    assert len(ncmapss.feature_sensors(include_virtual=True)) == 32


def test_invalid_split_raises():
    with pytest.raises(ValueError):
        ncmapss.load_ncmapss("train")  # N-CMAPSS uses dev/test, not train


def test_prepare_dataset_shapes_and_engine_disjoint():
    """LODO per-dataset prep: right window shapes + dev/test engines are disjoint (no leak)."""
    from pdm.data.cross_dataset import prepare_dataset

    d = prepare_dataset("DS02", paths.NCMAPSS_DS02, subsample=50, length=16, stride=16)
    F = len(ncmapss.XS_VARS)
    for k in ("devtrain", "devval", "test"):
        X, y, u, c = d[k]
        assert X.ndim == 3 and X.shape[1] == 16 and X.shape[2] == F
        assert len(y) == len(X) == len(u) == len(c) > 0
    dev_units = set(d["devtrain"][2]) | set(d["devval"][2])
    assert dev_units.isdisjoint(set(d["test"][2]))  # engine-wise integrity
