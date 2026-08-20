"""Build leak-safe interim Parquet tables from the raw C-MAPSS text files.

Adds the piecewise-linear ``rul`` label to every train/test row and writes one
parquet per split per sub-dataset into ``data/interim/``.

Run:  python -m pdm.data.build
"""
from __future__ import annotations

import pandas as pd

from pdm import paths
from pdm.data import cmapss


def build(cap: int | None = cmapss.DEFAULT_RUL_CAP) -> pd.DataFrame:
    paths.ensure_dirs()
    rows = []
    for sub in cmapss.SUBDATASETS:
        train = cmapss.add_train_rul(cmapss.load_raw(sub, "train"), cap=cap)
        rul_end = cmapss.load_test_rul(sub)
        test = cmapss.add_test_rul(cmapss.load_raw(sub, "test"), rul_end, cap=cap)

        train.to_parquet(paths.INTERIM / f"train_{sub}.parquet", index=False)
        test.to_parquet(paths.INTERIM / f"test_{sub}.parquet", index=False)

        rows.append(
            {
                "subdataset": sub,
                "train_units": train["unit"].nunique(),
                "train_rows": len(train),
                "test_units": test["unit"].nunique(),
                "test_rows": len(test),
                "rul_cap": int(train["rul"].max()),
            }
        )

    summary = pd.DataFrame(rows)
    print(f"Wrote interim parquet to {paths.INTERIM}\n")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    build()
