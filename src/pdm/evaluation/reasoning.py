"""
Per-engine maintenance reasoning — fuse the model outputs into a human verdict.

This is the layer a maintenance planner actually reads. For one engine it combines:

  * **RUL + calibrated 80% interval** (from the GRU + MC-dropout, M5) — *when*, with a
    confidence band,
  * **top degradation drivers** (from SHAP on the LightGBM baseline, M6) — *why*,
  * **input-drift status** (per-engine PSI vs the training distribution, M5) — *can we
    trust this prediction*,

into a single structured verdict with a plain-language headline. The division of labour
is deliberate: the deep model gives the calibrated number, the interpretable model
explains it — which is exactly why both locked models are carried forward.

The verdict ladder is driven by where the **whole 80% interval** sits relative to the
cost-tuned alert threshold ``T`` (recall ``lo <= mean <= hi``)::

    hi <= T            -> MAINTAIN_NOW   (even the optimistic bound is inside the window)
    mean <= T < hi     -> PLAN           (best estimate inside; upper tail still safe)
    lo <= T < mean     -> MONITOR        (only the pessimistic tail reaches the window)
    T < lo             -> HEALTHY        (entire interval clear of the window)

Pure functions only (no model objects), so the verdict logic is unit-testable; the
heavy model orchestration lives in ``notebooks/11_reasoning.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Action codes, most -> least urgent, with display metadata for the report/panel.
ACTIONS = {
    "MAINTAIN_NOW": {"urgency": "high", "icon": "🔴", "label": "Schedule maintenance now"},
    "PLAN": {"urgency": "medium", "icon": "🟠", "label": "Plan maintenance"},
    "MONITOR": {"urgency": "low", "icon": "🟡", "label": "Monitor closely"},
    "HEALTHY": {"urgency": "none", "icon": "🟢", "label": "Healthy — no action"},
}


def _action_for_interval(mean: float, lo: float, hi: float, threshold: float) -> str:
    if hi <= threshold:
        return "MAINTAIN_NOW"
    if mean <= threshold:
        return "PLAN"
    if lo <= threshold:
        return "MONITOR"
    return "HEALTHY"


def maintenance_verdict(
    unit: int,
    rul_mean: float,
    rul_lo: float,
    rul_hi: float,
    threshold: float,
    drivers: list[tuple[str, float, str]] | None = None,
    drift_psi: float | None = None,
    drift_flag: bool = False,
    drift_features: list[str] | None = None,
    lgbm_rul: float | None = None,
    horizon: float = 30.0,
) -> dict:
    """Fuse one engine's signals into a structured, human-readable maintenance verdict.

    Parameters mirror the fused signals: GRU calibrated interval (``rul_mean/lo/hi``),
    the cost-tuned alert ``threshold``, SHAP ``drivers`` (``[(sensor, contribution,
    label), ...]``), per-engine ``drift_*`` status, and an optional ``lgbm_rul``
    cross-check. Returns a dict with ``action``/``urgency``/``headline``/``reasons``/
    ``cautions`` plus the raw fields (JSON-friendly).
    """
    rul_mean, rul_lo, rul_hi = float(rul_mean), float(rul_lo), float(rul_hi)
    drivers = drivers or []
    action = _action_for_interval(rul_mean, rul_lo, rul_hi, threshold)
    meta = ACTIONS[action]

    ci = f"80% CI {rul_lo:.0f}–{rul_hi:.0f}"
    headline = (
        f"{meta['icon']} Engine {unit}: {meta['label'].lower()} — "
        f"RUL ≈ {rul_mean:.0f} cycles ({ci})."
    )

    reasons: list[str] = [f"Predicted remaining life ≈ {rul_mean:.0f} cycles ({ci})."]
    if action in ("MAINTAIN_NOW", "PLAN"):
        reasons.append(
            f"Best estimate is within the {threshold:.0f}-cycle alert window "
            f"(maintenance horizon {horizon:.0f})."
        )
    elif action == "MONITOR":
        reasons.append(
            f"Best estimate ({rul_mean:.0f}) is clear of the {threshold:.0f}-cycle window, "
            f"but the uncertainty band reaches into it ({rul_lo:.0f})."
        )
    else:
        reasons.append(
            f"Whole 80% interval stays above the {threshold:.0f}-cycle alert window."
        )

    if drivers:
        top = ", ".join(lab for _, _, lab in drivers[:3])
        verb = "pushing the estimate toward failure" if action != "HEALTHY" else "the largest degradation signals"
        reasons.append(f"Top degradation drivers ({verb}): {top}.")

    cautions: list[str] = []
    if lgbm_rul is not None:
        reasons.append(f"Cross-check: LightGBM baseline predicts {lgbm_rul:.0f} cycles.")
        disagree = abs(lgbm_rul - rul_mean)
        if disagree > max(15.0, 0.5 * rul_mean):
            cautions.append(
                f"Models disagree on RUL (GRU {rul_mean:.0f} vs LightGBM {lgbm_rul:.0f}, "
                f"Δ={disagree:.0f}) — treat the estimate with lower confidence."
            )
    if drift_flag:
        feat_txt = ""
        if drift_features:
            feat_txt = " (" + ", ".join(drift_features[:3]) + ")"
        psi_txt = f"PSI_max={drift_psi:.2f}" if drift_psi is not None else "elevated PSI"
        cautions.append(
            f"Sensor inputs for this engine differ from the training distribution "
            f"({psi_txt}){feat_txt} — prediction may be out-of-distribution; inspect sensors."
        )

    return {
        "unit": int(unit),
        "action": action,
        "urgency": meta["urgency"],
        "rul_mean": rul_mean,
        "rul_lo": rul_lo,
        "rul_hi": rul_hi,
        "threshold": float(threshold),
        "horizon": float(horizon),
        "lgbm_rul": None if lgbm_rul is None else float(lgbm_rul),
        "drift_psi": None if drift_psi is None else float(drift_psi),
        "drift_flag": bool(drift_flag),
        "top_drivers": [lab for _, _, lab in drivers[:3]],
        "headline": headline,
        "reasons": reasons,
        "cautions": cautions,
    }


def verdict_text(verdict: dict) -> str:
    """Render a verdict as a multi-line plain-language block (for logs / the report)."""
    lines = [verdict["headline"]]
    for r in verdict["reasons"]:
        lines.append(f"  • {r}")
    for c in verdict["cautions"]:
        lines.append(f"  ⚠ {c}")
    return "\n".join(lines)


# --------------------------------------------------------------------- drift helpers
def per_engine_psi(monitor, df_norm: pd.DataFrame) -> dict[int, dict]:
    """Score each engine's own (regime-normalized) trajectory against the reference.

    Returns ``{unit: {"psi_max", "psi_mean", "drifted_features"}}``. An engine is its
    own batch here, so PSI measures how far *that engine's* readings sit from the
    training population — a per-asset out-of-distribution signal.
    """
    out: dict[int, dict] = {}
    for unit, g in df_norm.groupby("unit", sort=False):
        s = monitor.score(g)
        out[int(unit)] = {
            "psi_max": s["psi_max"],
            "psi_mean": s["psi_mean"],
            "drifted_features": s["drifted_features"],
        }
    return out


def drift_baseline_threshold(monitor, train_norm: pd.DataFrame, units, q: float = 0.95) -> float:
    """Per-engine drift ceiling from in-distribution training engines.

    M5 lesson: the textbook PSI 0.25 is too sensitive for degradation-correlated
    sensors. Instead we take a high quantile of per-engine ``psi_max`` over known-good
    training engines as the "normal" ceiling, and flag deployment engines above it.
    """
    sub = train_norm[train_norm["unit"].isin(list(units))]
    psis = [monitor.score(g)["psi_max"] for _, g in sub.groupby("unit", sort=False)]
    return float(np.quantile(psis, q)) if psis else float("inf")
