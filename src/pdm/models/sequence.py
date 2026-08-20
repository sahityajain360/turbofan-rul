"""
Sequence deep-learning RUL models (LSTM / GRU / TCN) on regime-normalized telemetry.

Same data discipline as the baseline (engine-wise split, regime normalization,
piecewise RUL) and the **same evaluation** (NASA score + cost-aware decision), so deep
models compete head-to-head with the LightGBM baseline. ``predict_rul_seq`` mirrors the
baseline's ``predict_rul`` for train/serve parity.

Run:  python -m pdm.models.sequence --subdataset FD001 --model lstm
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pdm import paths
from pdm.data import cmapss
from pdm.data.sequences import make_last_windows, make_training_windows
from pdm.evaluation import CostModel, best_threshold, evaluate_decision, regression_report
from pdm.evaluation.metrics import rmse as _rmse
from pdm.features import RegimeNormalizer
from pdm.models.baseline import usable_sensors
from pdm.utils import seed_everything

WINDOW = 30
HORIZON = 30.0


# --------------------------------------------------------------------------- models
class RNNRegressor(nn.Module):
    def __init__(self, n_features: int, rnn: str = "lstm", hidden: int = 64,
                 layers: int = 2, dropout: float = 0.2):
        super().__init__()
        RNN = nn.LSTM if rnn == "lstm" else nn.GRU
        self.rnn = RNN(n_features, hidden, layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x):  # x: (B, T, F)
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class _Chomp(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, x):
        return x[:, :, : -self.size].contiguous() if self.size > 0 else x


class _TempBlock(nn.Module):
    def __init__(self, ci, co, k, dilation, dropout):
        super().__init__()
        pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(ci, co, k, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(co, co, k, padding=pad, dilation=dilation)
        self.chomp = _Chomp(pad)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(ci, co, 1) if ci != co else None

    def forward(self, x):
        y = self.drop(self.relu(self.chomp(self.conv1(x))))
        y = self.drop(self.relu(self.chomp(self.conv2(y))))
        res = x if self.down is None else self.down(x)
        return self.relu(y + res)


class TCNRegressor(nn.Module):
    def __init__(self, n_features: int, channels=(64, 64, 64), kernel: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        layers, ci = [], n_features
        for i, co in enumerate(channels):
            layers.append(_TempBlock(ci, co, kernel, 2**i, dropout))
            ci = co
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(ci, 1)

    def forward(self, x):  # (B, T, F) -> conv expects (B, F, T)
        y = self.tcn(x.transpose(1, 2))
        return self.head(y[:, :, -1]).squeeze(-1)


def default_config(model_type: str) -> dict:
    if model_type in ("lstm", "gru"):
        return {"hidden": 64, "layers": 2, "dropout": 0.2}
    if model_type == "tcn":
        return {"channels": [64, 64, 64], "kernel": 3, "dropout": 0.2}
    raise ValueError(f"unknown model_type {model_type!r}")


def build_model(model_type: str, n_features: int, config: dict) -> nn.Module:
    if model_type in ("lstm", "gru"):
        return RNNRegressor(n_features, rnn=model_type, **config)
    if model_type == "tcn":
        return TCNRegressor(n_features, **config)
    raise ValueError(f"unknown model_type {model_type!r}")


class AsymmetricMSE(nn.Module):
    """MSE that penalizes over-prediction (pred > true RUL — the risky direction) more
    heavily, aligning the training objective with the cost-asymmetric NASA score."""

    def __init__(self, over_penalty: float = 3.0):
        super().__init__()
        self.over = over_penalty

    def forward(self, pred, true):
        e = pred - true
        w = torch.where(
            e > 0,
            torch.as_tensor(self.over, device=e.device, dtype=e.dtype),
            torch.ones((), device=e.device, dtype=e.dtype),
        )
        return torch.mean(w * e * e)


# ----------------------------------------------------------------------- inference
def _predict_np(model: nn.Module, X: np.ndarray, device: str, batch: int = 2048) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i : i + batch]).to(device)
            outs.append(model(xb).cpu().numpy())
    return np.clip(np.concatenate(outs), 0, None)


def _build_loaded_model(bundle: dict, device: str) -> nn.Module:
    model = build_model(bundle["model_type"], len(bundle["sensors"]), bundle["config"]).to(device)
    model.load_state_dict(bundle["state_dict"])
    return model


def predict_rul_seq(bundle: dict, df_raw, device: str | None = None) -> np.ndarray:
    """Serving-parity inference: regime-normalize -> last window per engine -> predict."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    df = bundle["normalizer"].transform(df_raw)
    X, _ = make_last_windows(df, bundle["sensors"], bundle["window"])
    return _predict_np(_build_loaded_model(bundle, device), X, device)


def mc_predict(model: nn.Module, X: np.ndarray, n_samples: int = 50,
               device: str = "cuda", batch: int = 2048):
    """MC-dropout: ``n_samples`` stochastic passes with dropout active (our models have
    no BatchNorm, so train() == pure MC-dropout). Returns ``(mean, std, samples)``."""
    model.train()  # enable dropout for sampling
    outs = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds = []
            for i in range(0, len(X), batch):
                xb = torch.from_numpy(X[i : i + batch]).to(device)
                preds.append(model(xb).cpu().numpy())
            outs.append(np.clip(np.concatenate(preds), 0, None))
    S = np.stack(outs)
    return S.mean(0), S.std(0), S


def predict_rul_mc(bundle: dict, df_raw, n_samples: int = 50, device: str | None = None):
    """MC-dropout RUL with uncertainty: returns ``(mean, std, samples)`` per engine."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    df = bundle["normalizer"].transform(df_raw)
    X, _ = make_last_windows(df, bundle["sensors"], bundle["window"])
    return mc_predict(_build_loaded_model(bundle, device), X, n_samples, device)


# ------------------------------------------------------------------------ training
def train_sequence_model(
    subdataset: str = "FD001",
    model_type: str = "lstm",
    window: int = WINDOW,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 8,
    horizon: float = HORIZON,
    cap: int = cmapss.DEFAULT_RUL_CAP,
    seed: int = 42,
    n_regimes: int | None = None,
    loss: str = "mse",
    over_penalty: float = 3.0,
    save: bool = True,
    device: str | None = None,
    verbose: bool = True,
):
    seed_everything(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- data (regime-normalized, engine-wise split) -------------------------
    train_raw = cmapss.add_train_rul(cmapss.load_raw(subdataset, "train"), cap=cap)
    nr = cmapss.N_OPERATING_REGIMES[subdataset] if n_regimes is None else n_regimes
    normalizer = RegimeNormalizer(n_regimes=nr).fit(train_raw)
    train_norm = normalizer.transform(train_raw)
    sensors = usable_sensors(train_norm)

    tr_u, va_u = cmapss.split_units(train_norm["unit"].unique(), val_fraction=0.2, seed=seed)
    Xtr, ytr, _ = make_training_windows(train_norm[train_norm["unit"].isin(tr_u)], sensors, window)
    Xva, yva, _ = make_training_windows(train_norm[train_norm["unit"].isin(va_u)], sensors, window)

    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=batch_size, shuffle=True,
    )

    # --- model + optimization ------------------------------------------------
    config = default_config(model_type)
    model = build_model(model_type, len(sensors), config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = AsymmetricMSE(over_penalty) if loss == "asym" else nn.MSELoss()

    best_rmse, best_state, bad = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        va_rmse = _rmse(yva, _predict_np(model, Xva, device))
        if va_rmse < best_rmse - 1e-4:
            best_rmse, best_state, bad = va_rmse, deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if verbose and (epoch % 5 == 0 or bad >= patience):
            print(f"    [{model_type}/{subdataset}] epoch {epoch:02d} val_rmse={va_rmse:.2f} (best {best_rmse:.2f})")
        if bad >= patience:
            break
    model.load_state_dict(best_state)

    # --- evaluate (val for threshold, official test for reporting) -----------
    va_pred = _predict_np(model, Xva, device)
    cost = CostModel()
    best = best_threshold(yva, va_pred, horizon=horizon, cost=cost)

    test_norm = normalizer.transform(cmapss.load_raw(subdataset, "test"))
    Xte, _ = make_last_windows(test_norm, sensors, window)
    truth = cmapss.load_test_rul(subdataset)
    pred = _predict_np(model, Xte, device)

    bundle = {
        "model_type": model_type,
        "config": config,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "normalizer": normalizer,
        "sensors": sensors,
        "window": window,
        "rul_cap": cap,
        "horizon": horizon,
        "threshold": best["threshold"],
        "subdataset": subdataset,
    }
    metrics = {
        "subdataset": subdataset,
        "model_type": model_type,
        "window": window,
        "loss": loss,
        "n_regimes": nr,
        "n_sensors": len(sensors),
        "device": device,
        "epochs_run": epoch + 1,
        "val": regression_report(yva, va_pred),
        "test_raw_truth": regression_report(truth, pred),
        "test_capped_truth": regression_report(np.clip(truth, 0, cap), pred),
        "decision": evaluate_decision(truth, pred, horizon=horizon, threshold=best["threshold"], cost=cost),
        "threshold": best["threshold"],
        "cost_model": asdict(cost),
    }

    if save:
        torch.save(bundle, paths.MODELS / f"seq_{model_type}_{subdataset}.pt")
        (paths.REPORTS / f"seq_{model_type}_{subdataset}_metrics.json").write_text(
            json.dumps(metrics, indent=2)
        )
    return bundle, metrics, (truth, pred)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subdataset", default="FD001", choices=cmapss.SUBDATASETS)
    ap.add_argument("--model", default="lstm", choices=["lstm", "gru", "tcn"])
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    _, metrics, _ = train_sequence_model(args.subdataset, model_type=args.model, epochs=args.epochs)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
