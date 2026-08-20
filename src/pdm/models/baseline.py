"""
LightGBM RUL baseline + cost-aware maintenance decision, end to end.

Pipeline:
  1. drop constant sensors, build leak-safe rolling features,
  2. engine-wise train/val split,
  3. LightGBM regression on capped (piecewise-linear) RUL with early stopping,
  4. tune the alert threshold on validation by **cost** (not accuracy),
  5. evaluate on the official test set (per-engine last-cycle prediction),
  6. save a self-contained bundle for serving (train/serve parity via ``predict_rul``).

Run:  python -m pdm.models.baseline --subdataset FD001
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from pdm import paths
from pdm.data import cmapss
from pdm.evaluation import (
    CostModel,
    best_threshold,
    evaluate_decision,
    regression_report,
)
from pdm.features import (
    RegimeNormalizer,
    add_rolling_features,
    feature_columns,
    last_per_unit,
)
from pdm.utils import seed_everything

WINDOW = 30
HORIZON = 30.0
LOWVAR_STD = 1e-3  # sensors below this std on train are constant -> dropped


def usable_sensors(train: pd.DataFrame) -> list[str]:
    stds = train[cmapss.SENSOR_COLS].std()
    return [s for s in cmapss.SENSOR_COLS if stds[s] >= LOWVAR_STD]


def predict_rul(bundle: dict, df_raw: pd.DataFrame) -> np.ndarray:
    """Featurize raw cycles and predict RUL at each engine's last cycle.

    Shared by evaluation **and** the serving API so train and serve use the exact
    same feature path. Returns predictions in ascending unit order, clipped >= 0.
    """
    df = bundle["normalizer"].transform(df_raw)
    feats = add_rolling_features(df, bundle["sensors"], window=bundle["window"])
    last = last_per_unit(feats)
    X = last[feature_columns(bundle["sensors"])]
    return np.clip(bundle["model"].predict(X), 0, None)


def train_baseline(
    subdataset: str = "FD001",
    window: int = WINDOW,
    horizon: float = HORIZON,
    cap: int = cmapss.DEFAULT_RUL_CAP,
    seed: int = 42,
    n_regimes: int | None = None,
    save: bool = True,
):
    seed_everything(seed)
    paths.ensure_dirs()

    # --- regime normalization + features (leak-safe) --------------------------
    train_raw = cmapss.add_train_rul(cmapss.load_raw(subdataset, "train"), cap=cap)
    nr = cmapss.N_OPERATING_REGIMES[subdataset] if n_regimes is None else n_regimes
    normalizer = RegimeNormalizer(n_regimes=nr).fit(train_raw)
    train_norm = normalizer.transform(train_raw)
    sensors = usable_sensors(train_norm)  # regime-only sensors collapse -> dropped
    feat_cols = feature_columns(sensors)
    feats = add_rolling_features(train_norm, sensors, window=window)

    # --- engine-wise split ----------------------------------------------------
    tr_units, va_units = cmapss.split_units(feats["unit"].unique(), val_fraction=0.2, seed=seed)
    tr = feats[feats["unit"].isin(tr_units)]
    va = feats[feats["unit"].isin(va_units)]
    Xtr, ytr = tr[feat_cols], tr["rul"]
    Xva, yva = va[feat_cols], va["rul"]

    # --- model ----------------------------------------------------------------
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=30,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    va_pred = np.clip(model.predict(Xva), 0, None)
    val_report = regression_report(yva, va_pred)

    bundle = {
        "model": model,
        "normalizer": normalizer,
        "sensors": sensors,
        "window": window,
        "rul_cap": cap,
        "horizon": horizon,
        "subdataset": subdataset,
    }

    # --- tune alert threshold on validation, by cost --------------------------
    cost = CostModel()
    best = best_threshold(yva.to_numpy(), va_pred, horizon=horizon, cost=cost)
    bundle["threshold"] = best["threshold"]
    bundle["cost_model"] = asdict(cost)

    # --- official test evaluation (per-engine last cycle) ---------------------
    test_raw = cmapss.load_raw(subdataset, "test")
    truth = cmapss.load_test_rul(subdataset)  # raw RUL at each unit's last cycle
    pred = predict_rul(bundle, test_raw)

    metrics = {
        "subdataset": subdataset,
        "window": window,
        "n_regimes": nr,
        "n_features": len(feat_cols),
        "n_sensors": len(sensors),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or model.n_estimators),
        "val": val_report,
        "test_raw_truth": regression_report(truth, pred),
        "test_capped_truth": regression_report(np.clip(truth, 0, cap), pred),
        "decision": evaluate_decision(truth, pred, horizon=horizon, threshold=best["threshold"], cost=cost),
        "threshold": best["threshold"],
        "cost_model": asdict(cost),
    }

    if save:
        joblib.dump(bundle, paths.MODELS / f"baseline_{subdataset}.joblib")
        (paths.REPORTS / f"baseline_{subdataset}_metrics.json").write_text(
            json.dumps(metrics, indent=2)
        )
    return bundle, metrics, (truth, pred)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subdataset", default="FD001", choices=cmapss.SUBDATASETS)
    ap.add_argument("--window", type=int, default=WINDOW)
    args = ap.parse_args()
    _, metrics, _ = train_baseline(subdataset=args.subdataset, window=args.window)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
