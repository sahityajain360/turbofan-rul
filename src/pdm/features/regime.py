"""
Operating-regime normalization for multi-regime C-MAPSS sub-datasets.

In FD002/FD004 the engines cycle through six operating regimes (altitude, Mach,
throttle). Raw sensor values shift with the regime, which swamps the slow
degradation trend. We fix this by:

  1. clustering the operating settings into ``n_regimes`` regimes (KMeans),
  2. z-normalizing each sensor *within its regime* (mean/std learned on TRAIN only).

After this, a sensor that only moved because of regime switching collapses to ~0
variance (and gets dropped downstream), while genuine degradation is exposed on a
common scale across regimes. For single-regime sets (``n_regimes=1``) this is just
global standardization.

Fit on train, reuse on test/serving (the fitted object is saved in the model bundle).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from pdm.data.cmapss import OP_SETTING_COLS, SENSOR_COLS

# Cosmetic: tiny KMeans on op-settings triggers a benign MKL leak warning on Windows.
warnings.filterwarnings("ignore", message=".*KMeans is known to have a memory leak.*")


class RegimeNormalizer:
    def __init__(
        self,
        n_regimes: int = 6,
        sensors: list[str] | None = None,
        op_cols: list[str] | None = None,
        seed: int = 42,
        n_init: int = 10,
        max_fit_samples: int | None = None,
    ):
        self.n_regimes = int(n_regimes)
        self.sensors = list(sensors) if sensors is not None else list(SENSOR_COLS)
        self.op_cols = list(op_cols) if op_cols is not None else list(OP_SETTING_COLS)
        self.seed = seed
        # KMeans cost controls (matter for the millions of N-CMAPSS rows; defaults keep
        # the original C-MAPSS behavior). max_fit_samples fits centroids on a random row
        # subsample then assigns all rows — same centroids, far cheaper on huge inputs.
        self.n_init = n_init
        self.max_fit_samples = max_fit_samples

    def fit(self, df: pd.DataFrame) -> "RegimeNormalizer":
        if self.n_regimes > 1:
            self._op_scaler = StandardScaler().fit(df[self.op_cols])
            ops = self._op_scaler.transform(df[self.op_cols])
            fit_ops = ops
            if self.max_fit_samples is not None and len(ops) > self.max_fit_samples:
                rng = np.random.default_rng(self.seed)
                fit_ops = ops[rng.choice(len(ops), self.max_fit_samples, replace=False)]
            self._km = KMeans(
                n_clusters=self.n_regimes, random_state=self.seed, n_init=self.n_init
            ).fit(fit_ops)
            labels = self._km.predict(ops)  # == labels_ when fit on all rows
        else:
            self._op_scaler = None
            self._km = None
            labels = np.zeros(len(df), dtype=int)

        self._mean: dict[int, pd.Series] = {}
        self._std: dict[int, pd.Series] = {}
        sens = df[self.sensors]
        for r in range(self.n_regimes):
            seg = sens[labels == r]
            self._mean[r] = seg.mean()
            # avoid divide-by-zero for sensors constant within a regime
            self._std[r] = seg.std().replace(0.0, 1.0)
        return self

    def assign(self, df: pd.DataFrame) -> np.ndarray:
        """Regime label for each row (uses train-fitted centroids)."""
        if self._km is None:
            return np.zeros(len(df), dtype=int)
        return self._km.predict(self._op_scaler.transform(df[self.op_cols]))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.sensors] = out[self.sensors].astype(float)  # normalized values are float
        labels = self.assign(df)
        for r in range(self.n_regimes):
            mask = labels == r
            if not mask.any():
                continue
            out.loc[mask, self.sensors] = (
                out.loc[mask, self.sensors] - self._mean[r]
            ) / self._std[r]
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
