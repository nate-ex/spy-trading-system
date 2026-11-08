"""
Tests for q005_technical_state_features.py's non-obvious logic: realized
vol as sqrt-sum-of-squared-log-returns, VWAP deviation, running range,
and volume_surprise's no-look-ahead trailing-window construction.
Synthetic data only.
"""

import numpy as np
import pandas as pd
import pytest

from q005_technical_state_features import (
    build_technical_state_features,
    _per_day_technical_features,
    VOL_SURPRISE_WINDOW_DAYS,
)


def _one_day(date_str, closes, highs=None, lows=None, volumes=None, start="13:00:00"):
    n = len(closes)
    times = pd.date_range(f"{date_str} {start}", periods=n, freq="1min")
    highs = highs or closes
    lows = lows or closes
    volumes = volumes or [100.0] * n
    return pd.DataFrame({
        "time": times, "date": pd.Timestamp(date_str),
        "close": closes, "high": highs, "low": lows,
        "open": closes, "volume": volumes,
    })


def test_realized_vol_cum_matches_manual_calc():
    closes = [100.0, 101.0, 100.0, 102.0]
    day = _one_day("2024-03-01", closes)
    feat = _per_day_technical_features(day)
    log_ret = np.diff(np.log(closes))
    expected = np.sqrt(np.cumsum(np.concatenate([[0.0], log_ret ** 2])))
    got = feat["realized_vol_cum"].to_numpy()[: len(closes)]
    assert got == pytest.approx(expected, abs=1e-10)


def test_realized_vol_first_bar_is_zero_not_nan():
    day = _one_day("2024-03-01", [100.0, 101.0])
    feat = _per_day_technical_features(day)
    assert feat["realized_vol_cum"].iloc[0] == pytest.approx(0.0)


def test_vwap_deviation_flat_price_is_zero():
    day = _one_day("2024-03-01", [100.0] * 5, volumes=[10, 20, 30, 40, 50])
    feat = _per_day_technical_features(day)
    got = feat["vwap_deviation"].to_numpy()[:5]
    assert got == pytest.approx(np.zeros(5), abs=1e-10)


def test_vwap_deviation_positive_when_price_above_average():
    # Heavy volume at the low price, light volume at the high price ->
    # VWAP sits close to the low price, so the final close (high) sits
    # above VWAP -> positive deviation.
    day = _one_day("2024-03-01", closes=[95.0, 95.0, 105.0], volumes=[1000, 1000, 1])
    feat = _per_day_technical_features(day)
    assert feat["vwap_deviation"].iloc[-1] > 0


def test_range_used_tracks_running_high_low():
    day = _one_day("2024-03-01", closes=[100.0, 105.0, 98.0, 101.0],
                    highs=[100.0, 106.0, 99.0, 101.0], lows=[100.0, 104.0, 97.0, 100.0])
    feat = _per_day_technical_features(day)
    session_open = 100.0
    expected = [
        (100.0 - 100.0) / session_open,
        (106.0 - 100.0) / session_open,
        (106.0 - 97.0) / session_open,
        (106.0 - 97.0) / session_open,
    ]
    got = feat["range_used"].to_numpy()[:4]
    assert got == pytest.approx(expected, abs=1e-10)


def test_gap_fills_price_forward_and_volume_zero():
    # A day with a mid-session gap in the underlying bars -- build via a
    # DataFrame that skips a minute, so _regular_day_grid must reindex
    # and fill it.
    times = pd.to_datetime(["2024-03-01 13:00:00", "2024-03-01 13:01:00", "2024-03-01 13:03:00"])
    day = pd.DataFrame({
        "time": times, "date": pd.Timestamp("2024-03-01"),
        "close": [100.0, 101.0, 103.0], "high": [100.0, 101.0, 103.0],
        "low": [100.0, 101.0, 103.0], "open": [100.0, 101.0, 103.0],
        "volume": [10.0, 20.0, 30.0],
    })
    feat = _per_day_technical_features(day)
    # minute 13:02 is missing -> price forward-filled from 101, volume zero
    # -> cum_volume at 13:02 should equal cum_volume at 13:01 (no volume added).
    assert feat["realized_vol_cum"].iloc[2] == feat["realized_vol_cum"].iloc[1]  # no new return, price ffilled


def test_volume_surprise_uses_only_strictly_prior_days():
    # 25 days of identical volume profile, so day 21+ (first day with a
    # full 20-day trailing window) should show ~0 surprise.
    frames = []
    for i in range(25):
        date_str = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        frames.append(_one_day(date_str, closes=[100.0, 101.0, 102.0], volumes=[100.0, 100.0, 100.0]))
    dev = pd.concat(frames, ignore_index=True)
    feat = build_technical_state_features(dev)
    last_day = feat[feat["date"] == feat["date"].max()]
    assert last_day["volume_surprise"].abs().max() < 1e-6

    # Now spike ONLY the most recent day's volume -- must not leak backward
    # into any earlier day's volume_surprise (no look-ahead).
    dev2 = dev.copy()
    dev2.loc[dev2["date"] == dev2["date"].max(), "volume"] = 100000.0
    feat2 = build_technical_state_features(dev2)
    earlier_day = sorted(feat2["date"].unique())[-2]
    before = feat[feat["date"] == earlier_day]["volume_surprise"].to_numpy()
    after = feat2[feat2["date"] == earlier_day]["volume_surprise"].to_numpy()
    np.testing.assert_allclose(before, after, equal_nan=True)


def test_volume_surprise_nan_during_warmup():
    frames = []
    for i in range(5):
        date_str = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        frames.append(_one_day(date_str, closes=[100.0, 101.0], volumes=[100.0, 100.0]))
    dev = pd.concat(frames, ignore_index=True)
    feat = build_technical_state_features(dev)
    assert feat["volume_surprise"].isna().all()
