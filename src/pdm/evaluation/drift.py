"""
Input-drift monitoring (deployment-time non-stationarity guard).

A model is only trustworthy while its inputs resemble what it trained on. The
``DriftMonitor`` learns the reference distribution of each model-input feature on the
training data, then scores a new batch with the **Population Stability Index (PSI)**:

    PSI = sum_bins (q - p) * ln(q / p)

per feature (p = reference proportion, q = new proportion in the same bins). Rule of
thumb: PSI < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant drift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-6
SIGNIFICANT = 0.25


class DriftMonitor:
    def __init__(self, features: list[str], n_bins: int = 10):
        self.features = list(features)
        self.n_bins = n_bins

    @staticmethod
    def _proportions(counts: np.ndarray) -> np.ndarray:
        p = np.clip(counts.astype(float), _EPS, None)
        return p / p.sum()

    def fit(self, ref: pd.DataFrame) -> "DriftMonitor":
        self.edges_: dict[str, np.ndarray] = {}
        self.ref_prop_: dict[str, np.ndarray] = {}
        for f in self.features:
            x = ref[f].to_numpy()
            edges = np.unique(np.quantile(x, np.linspace(0, 1, self.n_bins + 1)))
            edges[0], edges[-1] = -np.inf, np.inf
            self.edges_[f] = edges
            counts, _ = np.histogram(x, bins=edges)
            self.ref_prop_[f] = self._proportions(counts)
        return self

    def psi(self, new: pd.DataFrame) -> dict[str, float]:
        out = {}
        for f in self.features:
            counts, _ = np.histogram(new[f].to_numpy(), bins=self.edges_[f])
            q = self._proportions(counts)
            p = self.ref_prop_[f]
            out[f] = float(np.sum((q - p) * np.log(q / p)))
        return out

    def score(self, new: pd.DataFrame) -> dict:
        psi = self.psi(new)
        vals = np.array(list(psi.values()))
        drifted = [f for f, v in psi.items() if v > SIGNIFICANT]
        return {
            "psi_per_feature": psi,
            "psi_mean": float(vals.mean()),
            "psi_max": float(vals.max()),
            "n_drifted": len(drifted),
            "drifted_features": drifted,
            "flag": bool(len(drifted) > 0),
        }


class MahalanobisOOD:
    """Out-of-distribution score via Mahalanobis distance to the training feature distribution.

    Unlike the per-feature PSI monitor (which compares *marginal* distributions), this uses the
    feature **covariance**, so it detects shifts in the *correlation structure* between sensors —
    which is where a novel fault MODE shows up after per-fleet normalization (M7.7c: PSI was blind
    to the failing DS04/DS08a fleets, this flags them as the top-2 OOD, r=0.89 vs LODO error). The
    deployment gate: high distance ⇒ the asset's degradation pattern is unlike training ⇒ withhold
    trust / fine-tune on target data instead of mispredicting.
    """

    def __init__(self, features: list[str]):
        self.features = list(features)

    def fit(self, ref: pd.DataFrame) -> "MahalanobisOOD":
        X = ref[self.features].to_numpy(float)
        self.mu_ = X.mean(axis=0)
        self.cov_inv_ = np.linalg.pinv(np.cov(X, rowvar=False))
        return self

    def distance(self, new: pd.DataFrame) -> np.ndarray:
        """Per-row Mahalanobis distance to the reference distribution."""
        d = new[self.features].to_numpy(float) - self.mu_
        return np.sqrt(np.clip(np.einsum("ij,jk,ik->i", d, self.cov_inv_, d), 0.0, None))

    def score(self, new: pd.DataFrame) -> dict:
        dist = self.distance(new)
        return {
            "maha_mean": float(dist.mean()),
            "maha_p90": float(np.percentile(dist, 90)),
            "maha_max": float(dist.max()),
        }
