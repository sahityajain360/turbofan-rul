"""
N-CMAPSS data layer: load the HDF5 run-to-failure dataset into tidy DataFrames.

N-CMAPSS (NASA PCoE #17, Arias Chao et al. 2021) simulates a fleet of turbofan engines
flown to failure under *real flight conditions*. Unlike the legacy C-MAPSS text files it
ships as HDF5 with a fixed, row-aligned layout and a **predefined dev/test split**:

  A_{split}    [N, 4]   auxiliary: unit, cycle, Fc (flight class), hs (health state)
  W_{split}    [N, 4]   flight / operating conditions: alt, Mach, TRA, T2
  X_s_{split}  [N, 14]  measured physical sensors (T24 .. Wf)
  X_v_{split}  [N, 14]  virtual sensors (model outputs; optional extra inputs)
  T_{split}    [N, 10]  TRUE health parameters (simulator ground truth — NOT an input)
  Y_{split}    [N, 1]   RUL target, in remaining flight cycles

The signals are 1 Hz across ~90-min flights -> millions of rows per sub-dataset, so the
loader **downsamples** with a strided read (every ``subsample``-th row) to stay tractable
on a laptop GPU. The same stride is applied to every aligned array so rows stay matched.

Guardrails carry over from C-MAPSS: split **by engine unit** (carve the dev units into
train/val), never use ``T`` (health params) as a feature — they are the hidden ground
truth and would leak — and keep the provided RUL as the label.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from pdm import paths

# --- schema: variable names in fixed column order (verified against DS02) ----
A_VARS = ["unit", "cycle", "Fc", "hs"]
W_VARS = ["alt", "Mach", "TRA", "T2"]
XS_VARS = ["T24", "T30", "T48", "T50", "P15", "P2", "P21", "P24", "Ps30",
           "P40", "P50", "Nf", "Nc", "Wf"]
XV_VARS = ["T40", "P30", "P45", "W21", "W22", "W25", "W31", "W32", "W48",
           "W50", "SmFan", "SmLPC", "SmHPC", "phi"]
T_VARS = ["fan_eff_mod", "fan_flow_mod", "LPC_eff_mod", "LPC_flow_mod", "HPC_eff_mod",
          "HPC_flow_mod", "HPT_eff_mod", "HPT_flow_mod", "LPT_eff_mod", "LPT_flow_mod"]

SPLITS = ("dev", "test")
DEFAULT_SUBSAMPLE = 10  # 1 Hz -> ~0.1 Hz; keeps DS02 dev ~0.5M rows

# Physical meaning of the operating conditions + measured sensors (Arias Chao 2021),
# for plain-language features / explanations (parity with cmapss.SENSOR_DESCRIPTIONS).
SENSOR_DESCRIPTIONS = {
    "alt": "altitude", "Mach": "flight Mach number", "TRA": "throttle resolver angle",
    "T2": "fan inlet total temperature",
    "T24": "LPC outlet temperature", "T30": "HPC outlet temperature",
    "T48": "HPT outlet temperature", "T50": "LPT outlet temperature",
    "P15": "bypass-duct total pressure", "P2": "fan inlet pressure",
    "P21": "fan outlet pressure", "P24": "LPC outlet pressure",
    "Ps30": "HPC outlet static pressure", "P40": "burner outlet pressure",
    "P50": "LPT outlet pressure", "Nf": "physical fan speed",
    "Nc": "physical core speed", "Wf": "fuel flow",
}


def feature_sensors(include_virtual: bool = False) -> list[str]:
    """Sensor/condition columns used as model inputs (operating conditions + X_s [+ X_v])."""
    return list(W_VARS) + list(XS_VARS) + (list(XV_VARS) if include_virtual else [])


def _resolve(path: Path | None) -> Path:
    path = Path(path) if path is not None else paths.NCMAPSS_DS02
    if not path.exists():
        raise FileNotFoundError(f"Missing N-CMAPSS file: {path}")
    return path


def _validate_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")


def unit_ids(split: str = "dev", path: Path | None = None) -> np.ndarray:
    """Sorted unique engine-unit ids for a split (reads only the unit column)."""
    _validate_split(split)
    with h5py.File(_resolve(path), "r") as f:
        return np.unique(f[f"A_{split}"][:, 0].astype(int))


def load_ncmapss(
    split: str = "dev",
    path: Path | None = None,
    subsample: int = DEFAULT_SUBSAMPLE,
    include_virtual: bool = False,
    include_health_params: bool = False,
) -> pd.DataFrame:
    """Load one split of an N-CMAPSS HDF5 file into a tidy, row-aligned DataFrame.

    Columns: ``unit, cycle, Fc, hs`` + operating conditions (W) + physical sensors (X_s)
    [+ virtual sensors X_v if requested] [+ health params T if requested] + ``rul``.
    Rows are decimated by ``subsample`` (strided read) to stay memory-tractable.
    """
    _validate_split(split)
    path = _resolve(path)
    if subsample < 1:
        raise ValueError("subsample must be >= 1")

    cols: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        def rd(name: str) -> np.ndarray:
            return np.asarray(f[name][::subsample])

        for arr, names in (
            (rd(f"A_{split}"), A_VARS),
            (rd(f"W_{split}"), W_VARS),
            (rd(f"X_s_{split}"), XS_VARS),
        ):
            for i, name in enumerate(names):
                cols[name] = arr[:, i]
        if include_virtual:
            xv = rd(f"X_v_{split}")
            for i, name in enumerate(XV_VARS):
                cols[name] = xv[:, i]
        if include_health_params:
            t = rd(f"T_{split}")
            for i, name in enumerate(T_VARS):
                cols[name] = t[:, i]
        rul = rd(f"Y_{split}").ravel()

    df = pd.DataFrame(cols)
    df["rul"] = rul.astype(int)
    for c in ("unit", "cycle", "Fc", "hs"):
        df[c] = df[c].astype(int)
    return df
