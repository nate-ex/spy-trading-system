"""
Tests for state_features_common.py: the shared trailing-percentile helper
(no-look-ahead, mirroring q005_technical_state_features.py's already-
tested volume_surprise pattern) and the expanding intraday z-score
(excludes the current reading from its own reference stats). Synthetic
data only.
"""

import numpy as np
import pandas as pd
import pytest

from state_features_common import (
    trailing_percentile_same_time_of_day, expanding_intraday_zscore, trailing_std_same_time_of_day,
    trailing_zscore_same_time_of_day,
)


def _one_day_grid(date_str, values, start="13:00:00"):
    n = len(values)
    times = pd.date_range(f"{date_str} {start}", periods=n, freq="1min")
    return pd.DataFrame({"time": times, "date": pd.Timestamp(date_str), "v": values})


def test_percentile_uses_only_strictly_prior_days():
    frames = []
    for i in range(25):
        date_str = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        frames.append(_one_day_grid(date_str, [float(i), float(i) + 1]))
    df = pd.concat(frames, ignore_index=True)
    out = trailing_percentile_same_time_of_day(df, "v", window_days=20, min_periods=10)

    df2 = df.copy()
    df2.loc[df2["date"] == df2["date"].max(), "v"] = 999999.0
    out2 = trailing_percentile_same_time_of_day(df2, "v", window_days=20, min_periods=10)

    earlier = sorted(df["date"].unique())[-2]
    before = out[out["date"] == earlier]["v_percentile"].to_numpy()
    after = out2[out2["date"] == earlier]["v_percentile"].to_numpy()
    np.testing.assert_allclose(before, after, equal_nan=True)


def test_percentile_matches_hand_computation():
    # Grid pos 0 monotonic increasing across 21 days: 0,1,...,20. The
    # last day's 20 strictly-prior values are all smaller -> percentile 1.0.
    frames = []
    for i in range(21):
        date_str = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        frames.append(_one_day_grid(date_str, [float(i)]))
    df = pd.concat(frames, ignore_index=True)
    out = trailing_percentile_same_time_of_day(df, "v", window_days=20, min_periods=10)
    last_day = df["date"].max()
    assert out[out["date"] == last_day]["v_percentile"].iloc[0] == pytest.approx(1.0)


def test_percentile_nan_during_warmup():
    frames = []
    for i in range(5):
        date_str = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        frames.append(_one_day_grid(date_str, [float(i)]))
    df = pd.concat(frames, ignore_index=True)
    out = trailing_percentile_same_time_of_day(df, "v", window_days=20, min_periods=10)
    assert out["v_percentile"].isna().all()


def test_expanding_zscore_excludes_current_reading():
    day_df = pd.DataFrame({"v": [1.0, 1.0, 1.0, 1.0, 100.0]})
    z = expanding_intraday_zscore(day_df, "v", min_periods=3)
    assert np.isnan(z[3])
    assert np.isnan(z[4])  # prior std (of four 1's) is 0 -> NaN, not a huge number


def test_expanding_zscore_hand_computed():
    day_df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 10.0]})
    z = expanding_intraday_zscore(day_df, "v", min_periods=2)
    prior_mean = np.mean([1.0, 2.0, 3.0])
    prior_std = np.std([1.0, 2.0, 3.0], ddof=0)
    expected = (10.0 - prior_mean) / prior_std
    assert z[3] == pytest.approx(expected)


def test_expanding_zscore_nan_before_min_periods():
    day_df = pd.DataFrame({"v": [1.0, 2.0]})
    z = expanding_intraday_zscore(day_df, "v", min_periods=3)
    assert np.isnan(z).all()


def test_trailing_std_excludes_current_day():
    frames = []
    for i in range(25):
        date_str = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        frames.append(_one_day_grid(date_str, [float(i), float(i) + 1]))
    df = pd.concat(frames, ignore_index=True)
    out = trailing_std_same_time_of_day(df, "v", window_days=20, min_periods=10)

    df2 = df.copy()
    df2.loc[df2["date"] == df2["date"].max(), "v"] = 999999.0
    out2 = trailing_std_same_time_of_day(df2, "v", window_days=20, min_periods=10)

    earlier = sorted(df["date"].unique())[-2]
    before = out[out["date"] == earlier]["v_tod_std"].to_numpy()
    after = out2[out2["date"] == earlier]["v_tod_std"].to_numpy()
    np.testing.assert_allclose(before, after, equal_nan=True)


def test_trailing_std_hand_computed():
    # Grid pos 0 across 21 days: 0,1,...,20. The last day's std should be
    # the population std (ddof=0) of the 20 strictly-prior values [0..19].
    frames = []
    for i in range(21):
        date_str = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        frames.append(_one_day_grid(date_str, [float(i)]))
    df = pd.concat(frames, ignore_index=True)
    out = trailing_std_same_time_of_day(df, "v", window_days=20, min_periods=10)
    last_day = df["date"].max()
    expected = np.std(np.arange(20, dtype=float), ddof=0)
    assert out[out["date"] == last_day]["v_tod_std"].iloc[0] == pytest.approx(expected)


def test_trailing_std_nan_during_warmup():
    frames = []
    for i in range(5):
        date_str = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        frames.append(_one_day_grid(date_str, [float(i)]))
    df = pd.concat(frames, ignore_index=True)
    out = trailing_std_same_time_of_day(df, "v", window_days=20, min_periods=10)
    assert out["v_tod_std"].isna().all()


def test_trailing_zscore_excludes_current_day():
    frames = []
    for i in range(25):
        date_str = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        frames.append(_one_day_grid(date_str, [float(i), float(i) + 1]))
    df = pd.concat(frames, ignore_index=True)
    out = trailing_zscore_same_time_of_day(df, "v", window_days=20, min_periods=10)

    df2 = df.copy()
    df2.loc[df2["date"] == df2["date"].max(), "v"] = 999999.0
    out2 = trailing_zscore_same_time_of_day(df2, "v", window_days=20, min_periods=10)

    earlier = sorted(df["date"].unique())[-2]
    before = out[out["date"] == earlier]["v_zscore_crossday"].to_numpy()
    after = out2[out2["date"] == earlier]["v_zscore_crossday"].to_numpy()
    np.testing.assert_allclose(before, after, equal_nan=True)


def test_trailing_zscore_hand_computed():
    # Grid pos 0 across 21 days: 0,1,...,20. The last day's z should use
    # the mean/std (ddof=0) of the 20 strictly-prior values [0..19].
    frames = []
    for i in range(21):
        date_str = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        frames.append(_one_day_grid(date_str, [float(i)]))
    df = pd.concat(frames, ignore_index=True)
    out = trailing_zscore_same_time_of_day(df, "v", window_days=20, min_periods=10)
    last_day = df["date"].max()
    prior = np.arange(20, dtype=float)
    expected = (20.0 - prior.mean()) / prior.std(ddof=0)
    assert out[out["date"] == last_day]["v_zscore_crossday"].iloc[0] == pytest.approx(expected)


def test_trailing_zscore_nan_during_warmup():
    frames = []
    for i in range(5):
        date_str = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        frames.append(_one_day_grid(date_str, [float(i)]))
    df = pd.concat(frames, ignore_index=True)
    out = trailing_zscore_same_time_of_day(df, "v", window_days=20, min_periods=10)
    assert out["v_zscore_crossday"].isna().all()


def test_trailing_zscore_nan_when_std_zero():
    frames = []
    for i in range(25):
        date_str = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        frames.append(_one_day_grid(date_str, [1.0]))  # constant -> std 0
    df = pd.concat(frames, ignore_index=True)
    out = trailing_zscore_same_time_of_day(df, "v", window_days=20, min_periods=10)
    last_day = df["date"].max()
    assert pd.isna(out[out["date"] == last_day]["v_zscore_crossday"].iloc[0])
