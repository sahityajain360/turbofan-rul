"""
SHAP explainability for the LightGBM RUL baseline.

TreeExplainer gives every prediction an exact additive decomposition::

    predicted_RUL = base_value + sum_f shap_value[f]

so we can read *why* an engine's RUL is what it is, not just the number. Each sensor
contributes four engineered features (mean / std / last / slope); because SHAP values
are additive we sum a sensor's four values into one **net signed contribution** in
cycles:

  * net < 0  -> the sensor is pushing predicted RUL *down* (toward failure) = a
    **degradation driver**,
  * net > 0  -> the sensor is signalling remaining health.

This module holds the model-agnostic SHAP math (so it is unit-testable without the
saved bundle) plus a thin :class:`RULExplainer` that wires it to a baseline bundle for
the notebooks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import shap

from pdm.data.cmapss import sensor_label
from pdm.features import add_rolling_features, feature_columns, last_per_unit

_STATS = ("mean", "std", "last", "slope")


def sensor_of(feature: str) -> str:
    """``"sensor_11_slope"`` -> ``"sensor_11"`` (strip the trailing stat name)."""
    return feature.rsplit("_", 1)[0]


def tree_shap(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Run TreeExplainer on a fitted tree model.

    Returns ``(values, base_value)`` where ``values`` is ``(n_rows, n_features)`` aligned
    to ``X``'s columns and ``base_value`` is the scalar model expectation. Guaranteed
    (TreeExplainer's additivity) so that ``base + values.sum(1) ~= model.predict(X)``.
    """
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(X), dtype=float)
    base = float(np.ravel(explainer.expected_value)[0])
    return values, base


def aggregate_to_sensors(
    values: np.ndarray, feat_cols: list[str], sensors: list[str]
) -> pd.DataFrame:
    """Sum per-feature SHAP values into per-sensor net contributions.

    ``values`` is ``(n_rows, n_features)`` aligned to ``feat_cols``. Returns a DataFrame
    ``(n_rows, n_sensors)`` of signed sums, columns ordered like ``sensors``.
    """
    df = pd.DataFrame(np.asarray(values, float), columns=list(feat_cols))
    out = {}
    for s in sensors:
        cols = [c for c in (f"{s}_{stat}" for stat in _STATS) if c in df.columns]
        out[s] = df[cols].sum(axis=1).to_numpy()
    return pd.DataFrame(out)


def global_sensor_importance(
    values: np.ndarray, feat_cols: list[str], sensors: list[str]
) -> tuple[dict[str, float], dict[str, float]]:
    """Mean ``|SHAP|`` importance per feature and per sensor, each sorted descending.

    Per-sensor importance sums the four stat-features' mean-absolute SHAP — i.e. the
    sensor's total influence on the prediction, regardless of direction.
    """
    mean_abs = np.abs(np.asarray(values, float)).mean(axis=0)
    per_feature = {f: float(v) for f, v in zip(feat_cols, mean_abs)}
    per_sensor: dict[str, float] = {}
    for f, v in per_feature.items():
        per_sensor[sensor_of(f)] = per_sensor.get(sensor_of(f), 0.0) + v
    per_feature = dict(sorted(per_feature.items(), key=lambda kv: kv[1], reverse=True))
    per_sensor = dict(sorted(per_sensor.items(), key=lambda kv: kv[1], reverse=True))
    return per_feature, per_sensor


@dataclass
class EngineDrivers:
    """Per-engine attribution: which sensors pushed RUL down (drivers) vs up (health)."""

    unit: int
    pred_rul: float  # SHAP-reconstructed raw model output (base + sum of contributions)
    base_value: float
    sensor_net: dict[str, float]  # sensor -> net signed contribution (cycles)
    degradation_drivers: list[tuple[str, float, str]] = field(default_factory=list)
    health_indicators: list[tuple[str, float, str]] = field(default_factory=list)

    def driver_labels(self, k: int | None = None) -> list[str]:
        items = self.degradation_drivers if k is None else self.degradation_drivers[:k]
        return [lab for _, _, lab in items]


class RULExplainer:
    """SHAP wrapper for a saved LightGBM baseline bundle (from ``train_baseline``)."""

    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.sensors = list(bundle["sensors"])
        self.feat_cols = feature_columns(self.sensors)
        self.explainer = shap.TreeExplainer(bundle["model"])
        self.base_value = float(np.ravel(self.explainer.expected_value)[0])

    def features_at_last_cycle(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Regime-normalize raw cycles -> rolling features -> last cycle per engine.

        Same feature path as ``predict_rul`` (train/serve parity), with ``unit``/``cycle``
        carried through for labelling.
        """
        df = self.bundle["normalizer"].transform(df_raw)
        feats = add_rolling_features(df, self.sensors, window=self.bundle["window"])
        return last_per_unit(feats)

    def shap_values(self, last_feats: pd.DataFrame) -> np.ndarray:
        """SHAP matrix ``(n_engines, n_features)`` for last-cycle feature rows."""
        return np.asarray(self.explainer.shap_values(last_feats[self.feat_cols]), float)

    def global_importance(
        self, last_feats: pd.DataFrame
    ) -> tuple[dict[str, float], dict[str, float]]:
        return global_sensor_importance(
            self.shap_values(last_feats), self.feat_cols, self.sensors
        )

    def per_engine(self, last_feats: pd.DataFrame, top_k: int = 4) -> list[EngineDrivers]:
        """Per-engine signed sensor attributions, ranked into drivers vs health signals."""
        values = self.shap_values(last_feats)
        sensor_df = aggregate_to_sensors(values, self.feat_cols, self.sensors)
        units = last_feats["unit"].to_numpy()
        reconstructed = self.base_value + values.sum(axis=1)

        results: list[EngineDrivers] = []
        for i, unit in enumerate(units):
            net = {s: float(v) for s, v in sensor_df.iloc[i].items()}
            ranked = sorted(net.items(), key=lambda kv: kv[1])  # most negative first
            drivers = [(s, c, sensor_label(s, short=True)) for s, c in ranked if c < 0][:top_k]
            health = [
                (s, c, sensor_label(s, short=True)) for s, c in reversed(ranked) if c > 0
            ][:top_k]
            results.append(
                EngineDrivers(
                    unit=int(unit),
                    pred_rul=float(reconstructed[i]),
                    base_value=self.base_value,
                    sensor_net=net,
                    degradation_drivers=drivers,
                    health_indicators=health,
                )
            )
        return results
