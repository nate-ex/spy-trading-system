"""Tests for stage1_density_engine.py — fit/freeze/load round-trip, valid
probabilities, and the regular-hours horizon censoring in predict()."""
from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd
import pytest

from stage1_density_engine import Stage1DensityEngine


def _synth_dev(n_days=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    d = pd.Timestamp("2023-01-03")
    made = 0
    while made < n_days:
        if d.weekday() < 5:
            # 14:00..16:00 inclusive, 1-min
            times = [d + pd.Timedelta(hours=14) + pd.Timedelta(minutes=i) for i in range(121)]
            mtc = np.array([120 - i for i in range(121)], float)
            rv = abs(rng.normal(0.004, 0.001))
            sig = 0.001 * (1 + mtc / 120)  # scale grows with time-to-close
            r5 = rng.normal(0, sig)
            rows.append(pd.DataFrame({
                "time": times, "date": d.normalize(),
                "minutes_to_close": mtc,
                "realized_vol_cum": rv, "realized_vol_30min": rv * 0.5,
                "vix_close_lagged": 15 + 5 * rng.random(),
                "return_5min": r5, "return_10min": r5 * 1.3,
                "return_15min": r5 * 1.6, "return_20min": r5 * 1.8, "return_30min": r5 * 2.0,
                "return_to_close": r5 * (1 + mtc / 60),
            }))
            made += 1
        d += pd.Timedelta(days=1)
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def engine():
    return Stage1DensityEngine().fit(_synth_dev())


def test_fit_populates_all_horizons(engine):
    assert set(engine.params["horizons"]) == {"5min", "10min", "15min", "20min", "30min", "to_close"}
    for h, p in engine.params["by_horizon"].items():
        assert p["skew_eta"] > 2 and -1 < p["skew_lambda"] < 1


def test_freeze_load_roundtrip(engine, tmp_path):
    path = tmp_path / "eng.json"
    engine.freeze(path)
    loaded = Stage1DensityEngine.load(path)
    a = engine.predict({"minutes_to_close": 60, "realized_vol_cum": 0.004,
                        "realized_vol_30min": 0.002, "vix_close_lagged": 16.0}, now=time(15, 0))
    b = loaded.predict({"minutes_to_close": 60, "realized_vol_cum": 0.004,
                        "realized_vol_30min": 0.002, "vix_close_lagged": 16.0}, now=time(15, 0))
    assert a.keys() == b.keys()
    assert abs(a["5min"]["P_tail"] - b["5min"]["P_tail"]) < 1e-12


def test_probabilities_valid_and_consistent(engine):
    pred = engine.predict({"minutes_to_close": 45, "realized_vol_cum": 0.005,
                           "realized_vol_30min": 0.002, "vix_close_lagged": 18.0},
                          now=time(15, 15), tau_bps=10)
    for h, d in pred.items():
        assert 0 <= d["P_upside"] <= 1 and 0 <= d["P_downside"] <= 1
        assert abs(d["P_tail"] - (d["P_upside"] + d["P_downside"])) < 1e-9
        assert d["sigma_bps"] > 0


def test_regular_hours_censoring_in_predict(engine):
    # at 15:50, only 5min (->15:55) and 10min (->16:00) and to_close are valid;
    # 15min/20min/30min (->16:05/16:10/16:20) must be censored
    pred = engine.predict({"minutes_to_close": 10, "realized_vol_cum": 0.004,
                           "realized_vol_30min": 0.002, "vix_close_lagged": 16.0}, now=time(15, 50))
    assert "15min" not in pred and "20min" not in pred and "30min" not in pred
    assert "5min" in pred and "10min" in pred and "to_close" in pred
    # at 15:56, 5min (->16:01) censored too
    pred2 = engine.predict({"minutes_to_close": 4, "realized_vol_cum": 0.004,
                            "realized_vol_30min": 0.002, "vix_close_lagged": 16.0}, now=time(15, 56))
    assert "5min" not in pred2 and "10min" not in pred2 and "to_close" in pred2
    # at 14:00 (mtc=120), every fixed horizon including 20min/30min is valid
    pred3 = engine.predict({"minutes_to_close": 120, "realized_vol_cum": 0.004,
                            "realized_vol_30min": 0.002, "vix_close_lagged": 16.0}, now=time(14, 0))
    assert "20min" in pred3 and "30min" in pred3


def test_predict_returns_empty_before_1400(engine):
    # fit() only trains on [14:00, 16:00] rows -- predict() must refuse to
    # extrapolate before that window rather than silently returning numbers
    # from an untrained region.
    pred = engine.predict({"minutes_to_close": 200, "realized_vol_cum": 0.004,
                           "realized_vol_30min": 0.002, "vix_close_lagged": 16.0}, now=time(13, 59))
    assert pred == {}
    pred_ok = engine.predict({"minutes_to_close": 120, "realized_vol_cum": 0.004,
                              "realized_vol_30min": 0.002, "vix_close_lagged": 16.0}, now=time(14, 0))
    assert "5min" in pred_ok


def test_ptail_increases_with_horizon(engine):
    pred = engine.predict({"minutes_to_close": 90, "realized_vol_cum": 0.004,
                           "realized_vol_30min": 0.002, "vix_close_lagged": 16.0},
                          now=time(14, 30), tau_bps=10)
    # longer horizon -> wider distribution -> larger P_tail at fixed tau
    assert (pred["5min"]["P_tail"] < pred["10min"]["P_tail"] < pred["15min"]["P_tail"]
           < pred["20min"]["P_tail"] < pred["30min"]["P_tail"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
