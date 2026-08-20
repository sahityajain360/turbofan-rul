"""
C-MAPSS data layer: parsing, RUL labeling, and engine-wise splitting.

This module is the single place where the dataset's structure and the project's
*non-negotiable guardrails* live:

  * RUL is **piecewise-linear** (clipped at ``rul_cap``) so the model is not asked
    to predict degradation that has not physically begun.
  * Train/validation splits are done **by engine unit**, never by random row,
    so a single engine's trajectory cannot leak across the split.
  * Test RUL is reconstructed from the per-unit end-of-trajectory RUL files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pdm import paths

# --- Schema (26 columns, fixed order) ---------------------------------------
INDEX_COLS = ["unit", "cycle"]
OP_SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]  # sensor_1 .. sensor_21
ALL_COLUMNS = INDEX_COLS + OP_SETTING_COLS + SENSOR_COLS  # len == 26

SUBDATASETS = ["FD001", "FD002", "FD003", "FD004"]
DEFAULT_RUL_CAP = 125

# Distinct operating regimes per sub-dataset (from the dataset spec, confirmed in EDA).
N_OPERATING_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}

# Physical meaning of each sensor channel (Saxena et al., PHM08 — the canonical
# C-MAPSS mapping). Used to turn "sensor_11" into plain language for explanations and
# the boss-facing report. Each value is ``(symbol, short description)``.
SENSOR_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "sensor_1": ("T2", "fan inlet temperature"),
    "sensor_2": ("T24", "LPC outlet temperature"),
    "sensor_3": ("T30", "HPC outlet temperature"),
    "sensor_4": ("T50", "LPT outlet temperature"),
    "sensor_5": ("P2", "fan inlet pressure"),
    "sensor_6": ("P15", "bypass-duct pressure"),
    "sensor_7": ("P30", "HPC outlet pressure"),
    "sensor_8": ("Nf", "physical fan speed"),
    "sensor_9": ("Nc", "physical core speed"),
    "sensor_10": ("epr", "engine pressure ratio"),
    "sensor_11": ("Ps30", "HPC outlet static pressure"),
    "sensor_12": ("phi", "fuel flow to Ps30 ratio"),
    "sensor_13": ("NRf", "corrected fan speed"),
    "sensor_14": ("NRc", "corrected core speed"),
    "sensor_15": ("BPR", "bypass ratio"),
    "sensor_16": ("farB", "burner fuel-air ratio"),
    "sensor_17": ("htBleed", "bleed enthalpy"),
    "sensor_18": ("Nf_dmd", "demanded fan speed"),
    "sensor_19": ("PCNfR_dmd", "demanded corrected fan speed"),
    "sensor_20": ("W31", "HPT coolant bleed flow"),
    "sensor_21": ("W32", "LPT coolant bleed flow"),
}


def sensor_label(sensor: str, short: bool = False) -> str:
    """Human-readable label for a sensor column.

    ``sensor_11`` -> ``"sensor_11 (Ps30: HPC outlet static pressure)"`` (or, with
    ``short=True``, ``"Ps30 (HPC outlet static pressure)"``). Unknown names pass through.
    """
    if sensor not in SENSOR_DESCRIPTIONS:
        return sensor
    sym, desc = SENSOR_DESCRIPTIONS[sensor]
    return f"{sym} ({desc})" if short else f"{sensor} ({sym}: {desc})"


def load_raw(subdataset: str, split: str, raw_dir: Path | None = None) -> pd.DataFrame:
    """Load a raw C-MAPSS file (e.g. ``train_FD001.txt``) into a tidy DataFrame.

    Parameters
    ----------
    subdataset : one of FD001..FD004
    split      : "train" or "test"
    """
    _validate(subdataset, split)
    raw_dir = raw_dir or paths.CMAPSS_RAW
    path = raw_dir / f"{split}_{subdataset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing C-MAPSS file: {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None)
    # Files have a trailing space -> a stray all-NaN 27th column; drop it.
    if df.shape[1] == len(ALL_COLUMNS) + 1:
        df = df.iloc[:, : len(ALL_COLUMNS)]
    if df.shape[1] != len(ALL_COLUMNS):
        raise ValueError(
            f"{path.name}: expected {len(ALL_COLUMNS)} columns, got {df.shape[1]}"
        )
    df.columns = ALL_COLUMNS
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df.sort_values(INDEX_COLS).reset_index(drop=True)


def load_test_rul(subdataset: str, raw_dir: Path | None = None) -> np.ndarray:
    """Load ``RUL_FDxxx.txt``: true RUL at each test unit's *last* recorded cycle.

    Returns a 1-D array indexed by ``unit - 1`` (unit ids are 1-based and contiguous).
    """
    _validate(subdataset, "test")
    raw_dir = raw_dir or paths.CMAPSS_RAW
    path = raw_dir / f"RUL_{subdataset}.txt"
    return pd.read_csv(path, header=None).iloc[:, 0].to_numpy().astype(int)


def add_train_rul(df: pd.DataFrame, cap: int | None = DEFAULT_RUL_CAP) -> pd.DataFrame:
    """Add a piecewise-linear ``rul`` column to a *train* frame (run-to-failure).

    For training engines the last cycle == failure, so RUL = max_cycle - cycle,
    optionally clipped at ``cap``.
    """
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    rul = max_cycle - out["cycle"]
    if cap is not None:
        rul = rul.clip(upper=cap)
    out["rul"] = rul.astype(int)
    return out


def add_test_rul(
    df: pd.DataFrame, rul_end: np.ndarray, cap: int | None = DEFAULT_RUL_CAP
) -> pd.DataFrame:
    """Add a piecewise-linear ``rul`` column to a *test* frame.

    Test trajectories stop *before* failure; ``rul_end[u-1]`` gives the true RUL
    at unit ``u``'s last cycle. So RUL(t) = (max_cycle_u - t) + rul_end[u-1].
    """
    out = df.copy()
    units = out["unit"].to_numpy()
    if units.max() > len(rul_end):
        raise ValueError("rul_end shorter than the number of test units")
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    rul_at_end = pd.Series(rul_end[units - 1], index=out.index)
    rul = (max_cycle - out["cycle"]) + rul_at_end
    if cap is not None:
        rul = rul.clip(upper=cap)
    out["rul"] = rul.astype(int)
    return out


def split_units(
    units, val_fraction: float = 0.2, seed: int = 42
) -> tuple[set[int], set[int]]:
    """Engine-wise train/val split. Returns ``(train_units, val_units)`` as id sets.

    Splitting on *units* (not rows) is the core anti-leakage guarantee.
    """
    u = np.array(sorted(set(int(x) for x in units)))
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    n_val = int(round(len(u) * val_fraction))
    val_units = set(u[:n_val].tolist())
    train_units = set(u[n_val:].tolist())
    return train_units, val_units


def _validate(subdataset: str, split: str) -> None:
    if subdataset not in SUBDATASETS:
        raise ValueError(f"subdataset must be one of {SUBDATASETS}, got {subdataset!r}")
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
