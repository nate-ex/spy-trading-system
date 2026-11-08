"""Tests for touch_probability_engine.py -- fit/freeze/load round-trip,
valid probabilities, and the regular-hours horizon censoring in predict().
Mirrors test_stage1_density_engine.py's structure/coverage for the
companion (touch/first-passage) engine, which previously had no dedicated
test file despite driving the dashboard's PRIMARY probability cards."""
from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd
import pytest

from touch_probability_engine import TouchProbabilityEngine


def _synth_dev(n_days=60, seed=0):
    """Synthetic max-excursion targets, scale growing with time-to-close --
    same shape convention as test_stage1_density_engine._synth_dev, but
    with the nonnegative max_excursion_{h} columns this engine's fit()
    needs instead of signed return_{h} columns."""
    rng = np.random.default_rng(seed)
    rows = []
    d = pd.Timestamp("2023-01-03")
    made = 0
    while made < n_days:
        if d.weekday() < 5:
            times = [d + pd.Timedelta(hours=14) + pd.Timedelta(minutes=i) for i in range(121)]
            mtc = np.array([120 - i for i in range(121)], float)
            rv = abs(rng.normal(0.004, 0.001))
            scale5 = 0.0008 * (1 + mtc / 120)
            exc5 = np.abs(rng.normal(0, scale5)) + 0.0001
            rows.append(pd.DataFrame({
                "time": times, "date": d.normalize(),
                "minutes_to_close": mtc,
                "realized_vol_cum": rv, "realized_vol_30min": rv * 0.5,
                "vix_close_lagged": 15 + 5 * rng.random(),
                "max_excursion_5min": exc5,
                "max_excursion_10min": exc5 * 1.3,
                "max_excursion_15min": exc5 * 1.6,
                "max_excursion_20min": exc5 * 1.85,
                "max_excursion_30min": exc5 * 2.2,
            }))
            made += 1
        d += pd.Timedelta(days=1)
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def engine():
    return TouchProbabilityEngine().fit(_synth_dev())


def test_fit_populates_all_horizons(engine):
    assert set(engine.params["horizons"]) == {"5min", "10min", "15min", "20min", "30min"}
    for h, p in engine.params["by_horizon"].items():
        assert p["lognorm_s"] > 0 and p["lognorm_scale"] > 0
        assert p["n_fit"] > 0


def test_freeze_load_roundtrip(engine, tmp_path):
    path = tmp_path / "eng.json"
    engine.freeze(path)
    loaded = TouchProbabilityEngine.load(path)
    state = {"minutes_to_close": 60.0, "realized_vol_cum": 0.004,
            "realized_vol_30min": 0.0015, "vix_close_lagged": 16.0}
    a = engine.predict(state, now=time(15, 0))
    b = loaded.predict(state, now=time(15, 0))
    assert a.keys() == b.keys()
    assert abs(a["5min"]["P_touch"] - b["5min"]["P_touch"]) < 1e-12


def test_probabilities_valid(engine):
    pred = engine.predict({"minutes_to_close": 45.0, "realized_vol_cum": 0.005,
                           "realized_vol_30min": 0.002, "vix_close_lagged": 18.0},
                          now=time(15, 15), tau_bps=10.0)
    for h, d in pred.items():
        assert 0.0 <= d["P_touch"] <= 1.0
        assert d["scale_bps"] > 0


def test_regular_hours_censoring_in_predict(engine):
    # at 15:50, only 5min (->15:55) and 10min (->16:00) are valid;
    # 15min/20min/30min (->16:05/16:10/16:20) censored
    pred = engine.predict({"minutes_to_close": 10.0, "realized_vol_cum": 0.004,
                           "realized_vol_30min": 0.002, "vix_close_lagged": 16.0}, now=time(15, 50))
    assert "15min" not in pred and "20min" not in pred and "30min" not in pred
    assert "5min" in pred and "10min" in pred
    # at 15:56, 5min (->16:01) censored too -- nothing valid left
    pred2 = engine.predict({"minutes_to_close": 4.0, "realized_vol_cum": 0.004,
                            "realized_vol_30min": 0.002, "vix_close_lagged": 16.0}, now=time(15, 56))
    assert pred2 == {}
    # at 14:00 (mtc=120), every fixed horizon including 20min/30min is valid
    pred3 = engine.predict({"minutes_to_close": 120.0, "realized_vol_cum": 0.004,
                            "realized_vol_30min": 0.002, "vix_close_lagged": 16.0}, now=time(14, 0))
    assert "20min" in pred3 and "30min" in pred3


def test_predict_returns_empty_before_1400(engine):
    # fit() only trains on [14:00, 16:00] rows -- predict() must refuse to
    # extrapolate before that window (the same discipline applied to
    # stage1_density_engine.py's predict()).
    pred = engine.predict({"minutes_to_close": 200.0, "realized_vol_cum": 0.004,
                           "realized_vol_30min": 0.002, "vix_close_lagged": 16.0}, now=time(13, 59))
    assert pred == {}
    pred_ok = engine.predict({"minutes_to_close": 120.0, "realized_vol_cum": 0.004,
                              "realized_vol_30min": 0.002, "vix_close_lagged": 16.0}, now=time(14, 0))
    assert "5min" in pred_ok


def test_p_touch_increases_with_horizon(engine):
    # longer horizon -> more opportunities to touch the threshold -> larger P_touch
    pred = engine.predict({"minutes_to_close": 90.0, "realized_vol_cum": 0.004,
                           "realized_vol_30min": 0.002, "vix_close_lagged": 16.0},
                          now=time(14, 30), tau_bps=10.0)
    assert (pred["5min"]["P_touch"] < pred["10min"]["P_touch"] < pred["15min"]["P_touch"]
           < pred["20min"]["P_touch"] < pred["30min"]["P_touch"])


def test_p_touch_increases_with_realized_vol(engine):
    # higher realized vol -> wider excursion scale -> larger P_touch at fixed tau
    low = engine.predict({"minutes_to_close": 60.0, "realized_vol_cum": 0.002,
                          "realized_vol_30min": 0.001, "vix_close_lagged": 14.0}, now=time(15, 0))
    high = engine.predict({"minutes_to_close": 60.0, "realized_vol_cum": 0.010,
                           "realized_vol_30min": 0.006, "vix_close_lagged": 25.0}, now=time(15, 0))
    assert low["5min"]["P_touch"] < high["5min"]["P_touch"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
