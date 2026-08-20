"""Tests for the per-engine maintenance reasoning verdict + drift helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pdm.evaluation.drift import DriftMonitor
from pdm.evaluation.reasoning import (
    drift_baseline_threshold,
    maintenance_verdict,
    per_engine_psi,
    verdict_text,
)

T = 39.0  # alert threshold used across the ladder tests


def _verdict(mean, lo, hi, **kw):
    return maintenance_verdict(unit=1, rul_mean=mean, rul_lo=lo, rul_hi=hi, threshold=T, **kw)


def test_verdict_ladder_maintain_now():
    # whole interval below threshold
    assert _verdict(20, 10, 30)["action"] == "MAINTAIN_NOW"


def test_verdict_ladder_plan():
    # mean below, upper tail above
    assert _verdict(35, 25, 50)["action"] == "PLAN"


def test_verdict_ladder_monitor():
    # only the lower tail reaches into the window
    assert _verdict(50, 30, 70)["action"] == "MONITOR"


def test_verdict_ladder_healthy():
    # entire interval clear of the window
    v = _verdict(90, 70, 110)
    assert v["action"] == "HEALTHY"
    assert v["urgency"] == "none"


def test_verdict_drivers_surface_in_reasons_and_fields():
    drivers = [("sensor_11", -8.0, "Ps30 (HPC outlet static pressure)"),
               ("sensor_4", -5.0, "T50 (LPT outlet temperature)")]
    v = _verdict(20, 12, 28, drivers=drivers)
    assert v["top_drivers"][0].startswith("Ps30")
    assert any("Ps30" in r for r in v["reasons"])


def test_verdict_drift_adds_caution():
    v = _verdict(20, 12, 28, drift_flag=True, drift_psi=1.9, drift_features=["sensor_7"])
    assert any("out-of-distribution" in c or "differ from the training" in c for c in v["cautions"])
    assert v["drift_flag"] is True


def test_verdict_model_disagreement_caution():
    # GRU says ~20, LightGBM says ~80 -> disagreement flagged
    v = _verdict(20, 12, 28, lgbm_rul=80.0)
    assert any("disagree" in c.lower() for c in v["cautions"])
    # close agreement -> no disagreement caution
    v2 = _verdict(20, 12, 28, lgbm_rul=22.0)
    assert not any("disagree" in c.lower() for c in v2["cautions"])


def test_verdict_text_is_multiline_string():
    txt = verdict_text(_verdict(20, 10, 30))
    assert "Engine 1" in txt and "\n" in txt


def test_per_engine_psi_and_baseline_threshold():
    rng = np.random.default_rng(1)
    feats = ["sensor_1", "sensor_2"]
    # reference + two "good" engines drawn from the same distribution
    ref = pd.DataFrame(rng.normal(size=(2000, 2)), columns=feats)
    good = pd.concat(
        [
            pd.DataFrame(
                {"unit": u, "sensor_1": rng.normal(size=300), "sensor_2": rng.normal(size=300)}
            )
            for u in (1, 2)
        ],
        ignore_index=True,
    )
    monitor = DriftMonitor(feats).fit(ref)

    thr = drift_baseline_threshold(monitor, good, units=[1, 2], q=0.95)
    assert thr > 0

    # an engine shifted far off the reference must exceed the in-distribution ceiling
    bad = pd.DataFrame({"unit": 3, "sensor_1": rng.normal(loc=6, size=300),
                        "sensor_2": rng.normal(loc=6, size=300)})
    psi = per_engine_psi(monitor, bad)
    assert psi[3]["psi_max"] > thr
