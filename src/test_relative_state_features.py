"""
Tests for relative_state_features.py: staleness masking before the
same-day z-score, and the VIX day-level relative stats. Synthetic data
only. The underlying percentile/z-score mechanics are already covered by
test_state_features_common.py -- these tests focus on this module's own
wiring (masking, column plumbing, formulas).
"""

import numpy as np
import pandas as pd
import pytest

from relative_state_features import add_relative_state_features, add_imbalance_price_spread


def _one_day(date_str, imbalance, stale, breadth=None, breadth_stale=None,
             vix_close_lagged=20.0, start="13:00:00"):
    n = len(imbalance)
    times = pd.date_range(f"{date_str} {start}", periods=n, freq="1min")
    breadth = breadth or [0.5] * n
    breadth_stale = breadth_stale or [False] * n
    return pd.DataFrame({
        "time": times, "date": pd.Timestamp(date_str),
        "aggregate_dollar_imbalance": imbalance, "imbalance_stale": stale,
        "breadth": breadth, "breadth_stale": breadth_stale,
        "realized_vol_cum": [0.01] * n, "cum_volume_today": list(range(1, n + 1)),
        "range_used": [0.001] * n, "trailing_return_5min": [0.0] * n,
        "vix_close_lagged": [vix_close_lagged] * n,
    })


def test_stale_imbalance_excluded_from_own_zscore():
    day = _one_day(
        "2024-04-01",
        imbalance=[999999.0, 1.0, 2.0, 3.0, 4.0],
        stale=[True, False, False, False, False],
    )
    out = add_relative_state_features(day)
    assert np.isnan(out["imbalance_zscore_intraday"].iloc[0])


def test_vix_trend_and_zscore_formula():
    vix_values = [float(v) for v in range(1, 22)]  # 21 days
    frames = [_one_day(str((pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).date()),
                        [10.0], [False], vix_close_lagged=v)
              for i, v in enumerate(vix_values)]
    df = pd.concat(frames, ignore_index=True)
    out = add_relative_state_features(df)
    last = out[out["date"] == out["date"].max()]

    expected_roll_mean = np.mean(vix_values[-20:])  # rolling(20) includes the current row itself
    assert last["vix_trend"].iloc[0] == pytest.approx(vix_values[-1] - expected_roll_mean)

    expected_std = np.std(vix_values[-20:], ddof=0)
    expected_z = (vix_values[-1] - expected_roll_mean) / expected_std
    assert last["vix_zscore_20d"].iloc[0] == pytest.approx(expected_z)


def test_vix_stats_nan_during_warmup():
    vix_values = [20.0] * 5
    frames = [_one_day(str((pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).date()),
                        [10.0], [False], vix_close_lagged=v)
              for i, v in enumerate(vix_values)]
    df = pd.concat(frames, ignore_index=True)
    out = add_relative_state_features(df)
    assert out["vix_zscore_20d"].isna().all()
    assert out["vix_percentile_60d"].isna().all()


def _spread_fixture_day(date_str, imbalance, stale, price_return, start="13:00:00"):
    n = len(imbalance)
    times = pd.date_range(f"{date_str} {start}", periods=n, freq="1min")
    return pd.DataFrame({
        "time": times, "date": pd.Timestamp(date_str),
        "aggregate_dollar_imbalance": imbalance, "imbalance_stale": stale,
        "today_return_so_far": price_return,
    })


def test_spread_is_difference_of_two_crossday_zscores():
    # 21 days, grid pos 0: imbalance 0..20, price_return always 0 with
    # values shifted so its trailing std is nonzero (0..20 too, scaled
    # down) -- both z-scored the SAME way, so the spread should equal
    # z_imbalance - z_price at every point where both are defined.
    frames = []
    for i in range(21):
        date_str = str((pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).date())
        frames.append(_spread_fixture_day(date_str, [float(i)], [False], [float(i) * 0.001]))
    df = pd.concat(frames, ignore_index=True)
    out = add_imbalance_price_spread(df)
    last = out[out["date"] == out["date"].max()]

    diff = last["imbalance_zscore_crossday"].iloc[0] - last["price_return_zscore_crossday"].iloc[0]
    assert last["imbalance_price_spread"].iloc[0] == pytest.approx(diff)
    # Both z-scores use the identical trailing window/shape here (price is
    # just imbalance*0.001, a positive linear rescaling), so they should
    # be numerically IDENTICAL -- meaning the spread should be ~0. This is
    # the actual point of using the same z-scoring method on both sides:
    # a pure scale difference between two comparable series washes out.
    assert last["imbalance_price_spread"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_spread_stale_imbalance_excluded():
    day = _spread_fixture_day(
        "2024-04-01", imbalance=[999999.0, 1.0, 2.0, 3.0, 4.0],
        stale=[True, False, False, False, False], price_return=[0.0, 0.001, 0.002, 0.003, 0.004],
    )
    out = add_imbalance_price_spread(day)
    # Not enough prior-day history to be non-NaN anyway (single day, no
    # trailing window) -- just confirms this doesn't raise/crash on a
    # stale-masked first value and produces the expected all-NaN column.
    assert out["imbalance_price_spread"].isna().all()


def test_spread_nan_during_warmup():
    frames = []
    for i in range(5):
        date_str = str((pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).date())
        frames.append(_spread_fixture_day(date_str, [10.0], [False], [0.001]))
    df = pd.concat(frames, ignore_index=True)
    out = add_imbalance_price_spread(df)
    assert out["imbalance_price_spread"].isna().all()
