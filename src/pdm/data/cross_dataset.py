"""
Cross-dataset pipeline for N-CMAPSS (DS01-DS08).
Handles load, regime-normalization, scaling, and windowing across multiple datasets
individually to prevent cross-dataset leakages.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pdm import paths
from pdm.data import cmapss, ncmapss
from pdm.features import RegimeNormalizer, make_within_flight_windows

SENSORS = ncmapss.XS_VARS
OP_COLS = ncmapss.W_VARS

# Healthy subsets for cross-dataset / leave-one-dataset-out work.
# DS08d is excluded — it is truncated (32 bytes of HDF5 metadata missing) in NASA's
# distribution and cannot be read; see PROJECT_LOG M7.6.
LODO_DATASETS = ["DS01", "DS02", "DS03", "DS04", "DS05", "DS06", "DS07", "DS08a", "DS08c"]


def prepare_dataset(
    ds_id: str,
    file_path: Path,
    subsample: int = 5,
    length: int = 512,
    stride: int = 256,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """All window sets for one dataset, normalized by THIS dataset's dev-train units only.

    Leak-safe: the regime-normalizer + scaler are fit on dev-train units; the test split's
    units never enter the fit. Returns ``(X, y, unit, cycle)`` tuples keyed by:
      ``devtrain`` — labeled windows from dev train units (pretrain X + fine-tune),
      ``devval``   — labeled windows from dev val units   (pooled early-stop signal),
      ``test``     — labeled windows from the official test split (held-out evaluation).
    """
    dev = ncmapss.load_ncmapss("dev", path=file_path, subsample=subsample)
    test = ncmapss.load_ncmapss("test", path=file_path, subsample=subsample)
    tr_u, va_u = cmapss.split_units(
        ncmapss.unit_ids("dev", path=file_path), val_fraction=0.34, seed=42
    )

    norm = RegimeNormalizer(n_regimes=20, sensors=SENSORS, op_cols=OP_COLS,
                            n_init=3, max_fit_samples=100_000)
    norm.fit(dev[dev.unit.isin(tr_u)])
    dev_n, test_n = norm.transform(dev), norm.transform(test)
    scaler = StandardScaler().fit(dev_n[dev_n.unit.isin(tr_u)][SENSORS])
    dev_n[SENSORS] = scaler.transform(dev_n[SENSORS])
    test_n[SENSORS] = scaler.transform(test_n[SENSORS])

    def win(df):
        return make_within_flight_windows(df, SENSORS, length, stride)

    return {
        "devtrain": win(dev_n[dev_n.unit.isin(tr_u)]),
        "devval": win(dev_n[dev_n.unit.isin(va_u)]),
        "test": win(test_n),
    }


def get_all_hdf5_files() -> dict[str, Path]:
    """Scan NCMAPSS_RAW directory and return a dict mapping dataset ID to file path.
    Example: {'DS01': Path('.../N-CMAPSS_DS01-005.h5'), ...}
    """
    raw_dir = paths.NCMAPSS_RAW
    files = list(raw_dir.glob("*.h5"))
    mapping = {}
    for f in files:
        name = f.name.upper()
        if "DS01" in name:
            mapping["DS01"] = f
        elif "DS02" in name:
            mapping["DS02"] = f
        elif "DS03" in name:
            mapping["DS03"] = f
        elif "DS04" in name:
            mapping["DS04"] = f
        elif "DS05" in name:
            mapping["DS05"] = f
        elif "DS06" in name:
            mapping["DS06"] = f
        elif "DS07" in name:
            mapping["DS07"] = f
        elif "DS08A" in name:
            mapping["DS08a"] = f
        elif "DS08C" in name:
            mapping["DS08c"] = f
        elif "DS08D" in name:
            mapping["DS08d"] = f
    return mapping


def load_and_normalize_single(
    ds_id: str,
    file_path: Path,
    subsample: int = 5,
    length: int = 64,
    stride: int = 32,
    exclude_val_units_from_pretrain: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a single dataset, fit and apply regime-normalization and scaling, and slice
    into within-flight windows. Leak-safe (scalers fitted only on dev-train units).
    """
    print(f"Processing dataset {ds_id} (subsample={subsample})...")
    
    # 1. Load dev split
    df = ncmapss.load_ncmapss("dev", path=file_path, subsample=subsample)
    
    # 2. Split units into train/val
    units = ncmapss.unit_ids("dev", path=file_path)
    tr_u, va_u = cmapss.split_units(units, val_fraction=0.34, seed=42)
    
    # 3. Fit RegimeNormalizer on train units only (cheap KMeans for the ~1M rows/dataset)
    norm = RegimeNormalizer(n_regimes=20, sensors=SENSORS, op_cols=OP_COLS,
                            n_init=3, max_fit_samples=100_000)
    train_df = df[df.unit.isin(tr_u)]
    norm.fit(train_df)
    
    # Transform whole df
    df_n = norm.transform(df)
    
    # 4. Fit StandardScaler on normalized train units only
    scaler = StandardScaler()
    scaler.fit(df_n[df_n.unit.isin(tr_u)][SENSORS])
    df_n[SENSORS] = scaler.transform(df_n[SENSORS])
    
    # 5. Extract windows
    # If we are preparing for pretraining, we want to pretrain on train units
    # (and optionally exclude validation units to prevent validation leakage during downstream training)
    target_units = tr_u if exclude_val_units_from_pretrain else units
    target_df = df_n[df_n.unit.isin(target_units)]
    
    X, y, u, c = make_within_flight_windows(target_df, SENSORS, length, stride)
    print(f"  -> Extracted {len(X)} windows from dataset {ds_id}")
    return X, y, u, c


def get_cross_dataset_windows(
    ds_ids: list[str] | None = None,
    subsample: int = 5,
    length: int = 64,
    stride: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Load, normalize, and combine within-flight windows from multiple datasets.
    Returns: (X_combined, y_combined)
    """
    mapping = get_all_hdf5_files()
    if ds_ids is None:
        ds_ids = sorted(list(mapping.keys()))
        
    Xs, ys = [], []
    for ds in ds_ids:
        if ds not in mapping:
            print(f"Warning: Dataset {ds} not found in raw files.")
            continue
        try:
            X, y, _, _ = load_and_normalize_single(ds, mapping[ds], subsample, length, stride)
            if len(X) > 0:
                Xs.append(X)
                ys.append(y)
        except Exception as e:
            print(f"Warning: Skipping dataset {ds} due to error: {e}")
            
    if not Xs:

        raise ValueError("No data windows extracted from any dataset.")
        
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)
