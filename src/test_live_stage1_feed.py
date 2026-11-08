"""Unit tests for the live Stage 1 feature computation (probability dashboard)."""

from __future__ import annotations

import sys
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_stage1_feed as lsf  # noqa: E402
from stage1_density_engine import Stage1DensityEngine, ARTIFACT_PATH  # noqa: E402

# Fitted engine artifacts and the development dataset are built from private
# data and are not included in this public repo; these tests run only when present.
needs_private_data = pytest.mark.skipif(
    not ARTIFACT_PATH.exists(), reason="fitted engine artifact / private data not present")


def _synthetic_day_bars(n_minutes=90, start=datetime(2026, 7, 21, 13, 0)):
    rng = np.random.default_rng(7)
    times = pd.date_range(start, periods=n_minutes, freq="1min")
    rets = rng.normal(0, 0.0005, n_minutes)
    close = 500.0 * np.cumprod(1 + rets)
    return pd.DataFrame({"time": times, "open": close, "high": close * 1.0005,
                         "low": close * 0.9995, "close": close, "volume": 1000.0})


def test_minutes_to_close_at_1400_is_120():
    now = datetime(2026, 7, 21, 14, 0)
    assert lsf.minutes_to_close(now) == pytest.approx(120.0)


def test_minutes_to_close_at_1558_is_2():
    now = datetime(2026, 7, 21, 15, 58)
    assert lsf.minutes_to_close(now) == pytest.approx(2.0)


def test_compute_live_features_empty_bars_raises():
    # NoLiveBarsYetError specifically -- this is the EXPECTED/transient
    # "no data yet" case (e.g. before market open) the dashboard shows a
    # calm "waiting" message for, not the generic Bloomberg-config error.
    with pytest.raises(lsf.NoLiveBarsYetError):
        lsf.compute_live_features(pd.DataFrame(), datetime(2026, 7, 21, 14, 0), vix_close_lagged=16.0)


def test_compute_live_features_before_first_bar_is_nan_not_raise():
    # The live desk's full-day grid (LIVE_GRID_START/END in
    # live_stage1_feed.py) means `compute_live_features` never sees an
    # empty index just because `now` precedes the day's first bar --
    # unlike the backtest's fixed 13:00-16:15 grid, the live grid always
    # has SOME row at/before any `now` within the same day. What must
    # still hold is that querying before real data exists doesn't
    # fabricate a "warmed up" reading: realized_vol_cum/_30min stay NaN
    # (see is_warmed_up), the same "not ready yet" signal used everywhere
    # else, rather than raising NoLiveBarsYetError.
    bars = _synthetic_day_bars()
    feat = lsf.compute_live_features(bars, datetime(2026, 7, 21, 12, 0), vix_close_lagged=16.0)
    assert np.isnan(feat["realized_vol_cum"])
    assert np.isnan(feat["realized_vol_30min"])
    assert not lsf.is_warmed_up(feat)


def test_compute_live_features_mixed_calendar_dates_raises():
    # A Bloomberg/xbbg response that mixes two dates (timezone glitch, stale
    # poll, etc.) must fail loudly rather than silently compute features
    # against a grid inferred from only the first row's date.
    bars = _synthetic_day_bars()
    bars = pd.concat([bars, _synthetic_day_bars(start=datetime(2026, 7, 22, 13, 0))], ignore_index=True)
    # Plain ValueError, NOT NoLiveBarsYetError -- this is a real data-
    # integrity problem, not the expected/transient "no bars yet" case.
    with pytest.raises(ValueError, match="multiple calendar dates") as exc_info:
        lsf.compute_live_features(bars, datetime(2026, 7, 22, 14, 0), vix_close_lagged=16.0)
    assert not isinstance(exc_info.value, lsf.NoLiveBarsYetError)


def test_compute_live_features_early_session_realized_vol_30min_is_nan():
    bars = _synthetic_day_bars()
    feat = lsf.compute_live_features(bars, datetime(2026, 7, 21, 13, 10), vix_close_lagged=16.0)
    assert np.isnan(feat["realized_vol_30min"])
    assert feat["realized_vol_cum"] >= 0


def test_compute_live_features_after_warmup_is_finite():
    bars = _synthetic_day_bars()
    feat = lsf.compute_live_features(bars, datetime(2026, 7, 21, 14, 0), vix_close_lagged=16.0)
    assert np.isfinite(feat["realized_vol_30min"])
    assert feat["vix_close_lagged"] == 16.0


def test_is_warmed_up_false_when_any_required_field_nan():
    assert lsf.is_warmed_up({"minutes_to_close": 100, "realized_vol_cum": 0.01,
                             "realized_vol_30min": float("nan"), "vix_close_lagged": 16.0}) is False
    assert lsf.is_warmed_up({"minutes_to_close": 100, "realized_vol_cum": 0.01,
                             "realized_vol_30min": 0.005, "vix_close_lagged": 16.0}) is True


@needs_private_data
def test_predict_live_matches_direct_engine_predict():
    engine = Stage1DensityEngine.load()
    features = {"time": datetime(2026, 7, 21, 15, 0), "minutes_to_close": 60.0,
                "realized_vol_cum": 0.004, "realized_vol_30min": 0.0015, "vix_close_lagged": 16.0}
    live_pred = lsf.predict_live(engine, features, tau_bps=10.0)
    direct_pred = engine.predict({k: v for k, v in features.items() if k != "time"},
                                 now=time(15, 0), tau_bps=10.0)
    assert live_pred == direct_pred


@needs_private_data
def test_validate_against_historical_zero_error_on_real_development_data():
    import data_access

    dev = data_access.load_development()
    dev["time"] = pd.to_datetime(dev["time"])
    dev["date"] = pd.to_datetime(dev["date"])
    engine = Stage1DensityEngine.load()
    sample_date = pd.to_datetime(dev["date"]).dt.date.iloc[len(dev) // 2]
    report = lsf.validate_against_historical(dev, sample_date, engine)
    assert report["n_checked"] > 0
    for err in report["max_abs_error"].values():
        assert err is not None and err < 1e-9


def test_load_vix_close_lagged_uses_bloomberg_when_available(monkeypatch):
    monkeypatch.setattr(lsf, "fetch_bloomberg_vix_close_lagged", lambda as_of=None: 18.5)
    assert lsf.load_vix_close_lagged() == 18.5


def test_load_vix_close_lagged_falls_back_to_csv_on_bloomberg_failure(monkeypatch, tmp_path):
    def boom(as_of=None):
        raise RuntimeError("Terminal not reachable")

    monkeypatch.setattr(lsf, "fetch_bloomberg_vix_close_lagged", boom)
    csv_path = tmp_path / "VIX_History.csv"
    csv_path.write_text("DATE,OPEN,HIGH,LOW,CLOSE\n07/14/2026,17.0,17.5,16.0,16.5\n")
    assert lsf.load_vix_close_lagged(csv_path=csv_path) == 16.5


def test_fetch_today_spy_bars_rejects_unknown_source():
    with pytest.raises(ValueError):
        lsf.fetch_today_spy_bars(source="not-a-real-source")


def test_fetch_today_spy_bars_auto_dispatches_to_bloomberg_when_it_succeeds(monkeypatch):
    called = {}

    def fake_fetch(self, ticker=None):
        called["ticker"] = ticker
        return pd.DataFrame({"time": [], "open": [], "high": [], "low": [], "close": [], "volume": []})

    monkeypatch.setattr(lsf.BloombergBarFeed, "fetch_today_bars", fake_fetch)
    bars, source_used = lsf.fetch_today_spy_bars(ticker="SPY")
    assert called["ticker"] == "SPY"
    assert source_used == "bloomberg"


def test_fetch_today_spy_bars_auto_falls_back_to_alpaca_when_bloomberg_fails(monkeypatch):
    def boom(self, ticker=None):
        raise RuntimeError("Terminal not reachable")

    called = {}

    def fake_alpaca_fetch(self, ticker=None):
        called["ticker"] = ticker
        return pd.DataFrame({"time": [], "open": [], "high": [], "low": [], "close": [], "volume": []})

    monkeypatch.setattr(lsf.BloombergBarFeed, "fetch_today_bars", boom)
    monkeypatch.setattr(lsf.AlpacaBarFeed, "fetch_today_bars", fake_alpaca_fetch)
    bars, source_used = lsf.fetch_today_spy_bars(ticker="SPY")
    assert called["ticker"] == "SPY"
    assert source_used == "alpaca"


def test_fetch_today_spy_bars_dispatches_to_alpaca_when_requested(monkeypatch):
    called = {}

    def fake_fetch(self, ticker=None):
        called["ticker"] = ticker
        return pd.DataFrame({"time": [], "open": [], "high": [], "low": [], "close": [], "volume": []})

    monkeypatch.setattr(lsf.AlpacaBarFeed, "fetch_today_bars", fake_fetch)
    bars, source_used = lsf.fetch_today_spy_bars(ticker="SPY", source="alpaca")
    assert called["ticker"] == "SPY"
    assert source_used == "alpaca"


def test_fetch_today_spy_bars_dispatches_to_massive_when_requested(monkeypatch):
    called = {}

    def fake_fetch(self):
        called["hit"] = True
        return pd.DataFrame({"time": [], "open": [], "high": [], "low": [], "close": [], "volume": []})

    monkeypatch.setattr(lsf.MassiveSpyBarFeed, "fetch_today_bars", fake_fetch)
    bars, source_used = lsf.fetch_today_spy_bars(ticker="SPY", source="massive")
    assert called.get("hit") is True
    assert source_used == "massive"


@needs_private_data
def test_validate_against_historical_missing_date_raises():
    import data_access
    from datetime import date as _date

    dev = data_access.load_development()
    dev["time"] = pd.to_datetime(dev["time"])
    dev["date"] = pd.to_datetime(dev["date"])
    engine = Stage1DensityEngine.load()
    with pytest.raises(ValueError):
        lsf.validate_against_historical(dev, _date(1999, 1, 1), engine)
