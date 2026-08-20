"""
Inspect an N-CMAPSS HDF5 file: list every dataset (shape/dtype), decode the variable
-name arrays, and summarize the dev/test split (units, rows, RUL range).

Run:  conda run -n rapgen_youtube python scripts/inspect_ncmapss.py [path-to.h5]
"""
from __future__ import annotations

import sys

import h5py
import numpy as np

DEFAULT = "data/raw/ncmapss/N-CMAPSS_DS02-006.h5"


def _decode_vars(arr) -> list[str]:
    flat = np.array(arr).ravel()
    out = []
    for v in flat:
        if isinstance(v, bytes):
            out.append(v.decode().strip())
        else:
            out.append(str(v).strip())
    return out


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    print(f"file: {path}\n")
    with h5py.File(path, "r") as f:
        print("=== all datasets (shape, dtype) ===")
        keys = sorted(f.keys())
        for k in keys:
            d = f[k]
            print(f"  {k:14s} shape={str(d.shape):20s} dtype={d.dtype}")

        print("\n=== variable-name arrays ===")
        for k in keys:
            if k.endswith("_var"):
                print(f"  {k:10s} = {_decode_vars(f[k])}")

        # auxiliary columns tell us unit/cycle/flight-class/health-state
        print("\n=== dev/test summary ===")
        for split in ("dev", "test"):
            akey, ykey = f"A_{split}", f"Y_{split}"
            if akey not in f:
                continue
            A = np.array(f[akey])
            avars = _decode_vars(f["A_var"]) if "A_var" in f else [f"c{i}" for i in range(A.shape[1])]
            units = A[:, avars.index("unit")] if "unit" in avars else A[:, 0]
            n_units = len(np.unique(units))
            line = f"  {split:4s}: rows={A.shape[0]:>9,}  units={n_units}"
            if ykey in f:
                y = np.array(f[ykey]).ravel()
                line += f"  RUL[min/max]={y.min():.0f}/{y.max():.0f}"
            print(line)
            print(f"        unit ids: {sorted(np.unique(units).astype(int).tolist())}")


if __name__ == "__main__":
    main()
