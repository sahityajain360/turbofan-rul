"""
Per-flight-cycle reduction for N-CMAPSS.

N-CMAPSS gives ~11 k 1 Hz samples per flight but **one RUL label per flight**. We collapse
each flight's within-flight samples into a single summary row — the per-flight *mean* of
each (regime-normalized) sensor — so the reduced frame is structurally identical to a
C-MAPSS per-cycle frame (one row per ``(unit, cycle)`` with sensor columns + ``rul``).

That lets the whole C-MAPSS pipeline apply unchanged: ``add_rolling_features`` (rolling
stats over *flights*) feeds the LightGBM baseline, and ``make_flight_windows`` builds
flight-history sequences for the GRU. The per-flight mean is the N-CMAPSS analogue of a
C-MAPSS per-cycle sensor reading.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def reduce_to_flights(df: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """Collapse within-flight samples to one row per ``(unit, cycle)``.

    Each sensor column becomes its per-flight mean; ``rul`` (constant within a flight) is
    carried through. Returns a frame sorted by ``(unit, cycle)`` — same shape as a C-MAPSS
    per-cycle table, so downstream feature/sequence code is reused as-is.
    """
    g = df.groupby(["unit", "cycle"], sort=True)
    out = g[sensors].mean()
    out["rul"] = g["rul"].first()
    return out.reset_index()


def flight_summary(
    df: pd.DataFrame,
    sensors: list[str],
    stats: tuple[str, ...] = ("mean", "std", "min", "max"),
) -> tuple[pd.DataFrame, list[str]]:
    """Richer per-flight reduction (M7.2): one row per ``(unit, cycle)`` with a column per
    ``{sensor}_{stat}``, recovering within-flight distribution info the bare mean discards.

    Returns ``(frame, channel_cols)`` sorted by ``(unit, cycle)`` with ``rul`` carried through.
    """
    g = df.groupby(["unit", "cycle"], sort=True)
    agg = g[sensors].agg(list(stats))
    agg.columns = [f"{s}_{st}" for s, st in agg.columns]
    agg = agg.fillna(0.0)  # std is undefined for a 1-sample flight (shouldn't occur)
    channels = list(agg.columns)
    agg["rul"] = g["rul"].first()
    return agg.reset_index(), channels


def add_flight_deltas(
    flights: pd.DataFrame, cols: list[str], lag: int = 1
) -> tuple[pd.DataFrame, list[str]]:
    """Append flight-to-flight deltas ``{col}_d{lag} = col - col.shift(lag)`` per engine.

    Degradation is about *change* over flights, so these velocity channels add signal beyond
    the level. Leak-safe (trailing diff); the first ``lag`` flights per unit get 0.
    Returns ``(frame, new_cols)``.
    """
    out = flights.copy()
    new: list[str] = []
    for c in cols:
        name = f"{c}_d{lag}"
        out[name] = out.groupby("unit", sort=False)[c].diff(lag).fillna(0.0)
        new.append(name)
    return out, new


def make_within_flight_windows(
    df: pd.DataFrame, sensors: list[str], length: int = 64, stride: int = 32
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sliding sub-windows of the raw within-flight 1 Hz stream (M7.5).

    Each flight ``(unit, cycle)`` is sliced into length-``length`` windows (step ``stride``)
    over its in-order samples, never crossing a flight boundary. Every window inherits its
    flight's RUL, so one flight yields many windows — abundant data for self-supervised
    pretraining, and a label-augmenting view for fine-tuning. Returns
    ``(X[N, length, F], rul[N], unit[N], cycle[N])``; flights shorter than ``length`` are
    skipped (none in DS02 at the default subsample).
    """
    Xs, ys, us, cs = [], [], [], []
    for (unit, cycle), g in df.groupby(["unit", "cycle"], sort=True):
        vals = g[sensors].to_numpy(dtype=np.float32)  # rows are in flight-time order
        n = len(vals)
        if n < length:
            continue
        # vectorized sliding windows: [n-length+1, F, length] -> stride -> [nw, length, F]
        sw = sliding_window_view(vals, window_shape=length, axis=0)[::stride]
        sw = np.ascontiguousarray(sw.transpose(0, 2, 1))
        nw = sw.shape[0]
        Xs.append(sw)
        ys.append(np.full(nw, g["rul"].iloc[0], dtype=np.float32))
        us.append(np.full(nw, int(unit), dtype=int))
        cs.append(np.full(nw, int(cycle), dtype=int))
    if not Xs:
        F = len(sensors)
        return (np.empty((0, length, F), np.float32), np.empty((0,), np.float32),
                np.empty((0,), int), np.empty((0,), int))
    return (np.concatenate(Xs), np.concatenate(ys),
            np.concatenate(us), np.concatenate(cs))


def make_flight_windows(
    flights: pd.DataFrame, sensors: list[str], window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Left-padded window of the last ``window`` flights ending at **every** flight.

    Unlike ``make_last_windows`` (last flight only) this yields one window per flight, so
    every labeled flight-cycle gets a prediction — the N-CMAPSS evaluation convention.
    Engines shorter than ``window`` at a given position are left-padded by repeating the
    earliest available flight. Returns ``(X[N, window, F], y[N], units[N], cycles[N])`` in
    ascending ``(unit, cycle)`` order (matching ``reduce_to_flights``).
    """
    Xs, ys, us, cs = [], [], [], []
    for unit in sorted(flights["unit"].unique()):
        g = flights[flights["unit"] == unit].sort_values("cycle")
        vals = g[sensors].to_numpy(dtype=np.float32)
        ruls = g["rul"].to_numpy(dtype=np.float32)
        cycs = g["cycle"].to_numpy()
        n = len(g)
        for end in range(1, n + 1):
            start = max(0, end - window)
            win = vals[start:end]
            if len(win) < window:  # left-pad with the earliest available flight
                pad = np.repeat(win[:1], window - len(win), axis=0)
                win = np.concatenate([pad, win], axis=0)
            Xs.append(win)
            ys.append(ruls[end - 1])
            us.append(int(unit))
            cs.append(int(cycs[end - 1]))
    return (
        np.stack(Xs),
        np.asarray(ys, np.float32),
        np.asarray(us, int),
        np.asarray(cs, int),
    )
