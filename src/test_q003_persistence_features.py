"""
Tests for q003_persistence_features.py's non-obvious logic: the breadth
aggregation (reused build_aggregate_series with an indicator value_col),
and the per-day dynamics state machine (activation vs. genuine flip,
stale-lookback exclusion for slope/agreement). Synthetic data only.
"""

import numpy as np
import pandas as pd
import pytest

from q003_persistence_features import (
    build_breadth_series,
    add_dynamics_features,
    DAY_GRID_START,
    DAY_GRID_END,
)


def _raw_ticks(rows):
    """rows: list of (Symbol, Side, SaleTime_str, Imbalance, LastSale)."""
    df = pd.DataFrame(rows, columns=["Symbol", "Side", "SaleTime", "Imbalance", "LastSale"])
    df["SaleTime"] = pd.to_datetime("2024-03-01 " + df["SaleTime"])
    df["dollar_imbalance"] = df["Imbalance"] * df["Side"] * df["LastSale"]
    return df


def test_breadth_all_agree_is_one():
    imb = _raw_ticks([
        ("AAA", 1, "14:00:00", 100, 10.0),
        ("BBB", 1, "14:00:05", 100, 10.0),
        ("CCC", 1, "14:00:10", 100, 10.0),
    ])
    events = build_breadth_series(imb)
    last = events.iloc[-1]
    assert last["breadth"] == pytest.approx(1.0)
    assert last["n_positive"] == 3
    assert last["n_negative"] == 0


def test_breadth_reflects_majority_agreement():
    imb = _raw_ticks([
        ("AAA", 1, "14:00:00", 100, 10.0),   # aggregate positive so far
        ("BBB", 1, "14:00:05", 100, 10.0),
        ("CCC", -1, "14:00:10", 50, 10.0),   # doesn't flip the aggregate sign
    ])
    events = build_breadth_series(imb)
    last = events.iloc[-1]
    # aggregate = +1000 + 1000 - 500 = +1500 > 0 -> breadth uses n_positive
    assert last["aggregate_dollar_imbalance"] == pytest.approx(1500.0)
    assert last["breadth"] == pytest.approx(2 / 3)


def test_breadth_updates_when_symbol_flips_side():
    imb = _raw_ticks([
        ("AAA", 1, "14:00:00", 100, 10.0),
        ("BBB", -1, "14:00:05", 100, 10.0),
        ("AAA", -1, "14:00:10", 100, 10.0),  # AAA flips from + to -
    ])
    events = build_breadth_series(imb)
    last = events.iloc[-1]
    # both AAA and BBB are now negative -> aggregate negative, breadth = n_negative/2 = 1.0
    assert last["aggregate_dollar_imbalance"] < 0
    assert last["breadth"] == pytest.approx(1.0)
    assert last["n_negative"] == 2
    assert last["n_positive"] == 0


def _dev_day(date_str, rows):
    """rows: list of (time_str, aggregate_dollar_imbalance, imbalance_stale, close)."""
    df = pd.DataFrame(rows, columns=["time", "aggregate_dollar_imbalance", "imbalance_stale", "close"])
    df["time"] = pd.to_datetime(f"{date_str} " + df["time"])
    df["date"] = pd.Timestamp(date_str).date()
    df["return_to_close"] = 0.0
    return df


def test_first_live_reading_is_activation_not_genuine_flip():
    day = _dev_day("2024-03-01", [
        ("13:00:00", 100.0, True, 100.0),
        ("14:00:00", 500.0, False, 100.0),  # first live reading of the day
        ("14:01:00", 500.0, False, 100.0),
    ])
    feat = add_dynamics_features(day)
    first_live = feat[feat["time"] == pd.Timestamp("2024-03-01 14:00:00")].iloc[0]
    assert first_live["is_day_activation"]
    assert not first_live["is_genuine_flip"]
    assert first_live["minutes_since_flip"] == 0


def test_genuine_sign_reversal_is_flagged_and_resets_duration():
    day = _dev_day("2024-03-01", [
        ("14:00:00", 500.0, False, 100.0),
        ("14:01:00", 500.0, False, 100.0),
        ("14:02:00", 500.0, False, 100.0),
        ("14:03:00", -300.0, False, 100.0),  # genuine reversal
        ("14:04:00", -300.0, False, 100.0),
    ])
    feat = add_dynamics_features(day)
    flip_row = feat[feat["time"] == pd.Timestamp("2024-03-01 14:03:00")].iloc[0]
    assert flip_row["is_genuine_flip"]
    assert not flip_row["is_day_activation"]
    assert flip_row["minutes_since_flip"] == 0

    after_flip = feat[feat["time"] == pd.Timestamp("2024-03-01 14:04:00")].iloc[0]
    assert after_flip["minutes_since_flip"] == 1

    before_flip = feat[feat["time"] == pd.Timestamp("2024-03-01 14:02:00")].iloc[0]
    assert before_flip["minutes_since_flip"] == 2  # 3 live readings in, 0-indexed from activation


def test_stale_rows_produce_nan_sign_and_break_the_run():
    day = _dev_day("2024-03-01", [
        ("14:00:00", 500.0, False, 100.0),
        ("14:01:00", 500.0, True, 100.0),   # stale -- shouldn't count as live continuation
        ("14:02:00", 500.0, False, 100.0),
    ])
    feat = add_dynamics_features(day)
    stale_row = feat[feat["time"] == pd.Timestamp("2024-03-01 14:01:00")].iloc[0]
    assert np.isnan(stale_row["sign"])
    assert np.isnan(stale_row["minutes_since_flip"])


def test_slope_is_nan_when_lookback_point_is_stale():
    rows = [("14:00:00", 100.0, True, 100.0)]  # stale lookback point at t-5
    for m in range(1, 6):
        rows.append((f"14:0{m}:00", 100.0 + m * 10, False, 100.0))
    day = _dev_day("2024-03-01", rows)
    feat = add_dynamics_features(day)
    row_5min = feat[feat["time"] == pd.Timestamp("2024-03-01 14:05:00")].iloc[0]
    # lookback to 14:00 (5 min prior) is stale -> slope_5min must be NaN, not computed against a stale anchor
    assert np.isnan(row_5min["slope_5min"])


def test_slope_computed_correctly_when_both_points_live():
    rows = [("14:00:00", 100.0, False, 100.0)]
    for m in range(1, 6):
        rows.append((f"14:0{m}:00", 100.0 + m * 10, False, 100.0))
    day = _dev_day("2024-03-01", rows)
    feat = add_dynamics_features(day)
    row_5min = feat[feat["time"] == pd.Timestamp("2024-03-01 14:05:00")].iloc[0]
    # value at 14:00 = 100, at 14:05 = 150 -> slope = (150-100)/5 = 10
    assert row_5min["slope_5min"] == pytest.approx(10.0)


def test_agreement_score_counts_matching_windows():
    # Build 31 minutes of a constant positive sign -> every lookback (5/15/30) agrees
    rows = [(f"14:{m:02d}:00", 500.0, False, 100.0) for m in range(31)]
    day = _dev_day("2024-03-01", rows)
    feat = add_dynamics_features(day)
    last_row = feat[feat["time"] == pd.Timestamp("2024-03-01 14:30:00")].iloc[0]
    assert last_row["agree_5min"] == 1.0
    assert last_row["agree_15min"] == 1.0
    assert last_row["agree_30min"] == 1.0
    assert last_row["agreement_score"] == 3.0


def test_minutes_to_close_counts_down_to_1600():
    day = _dev_day("2024-03-01", [
        ("14:00:00", 500.0, False, 100.0),
        ("15:45:00", 500.0, False, 100.0),
    ])
    feat = add_dynamics_features(day)
    row_1400 = feat[feat["time"] == pd.Timestamp("2024-03-01 14:00:00")].iloc[0]
    row_1545 = feat[feat["time"] == pd.Timestamp("2024-03-01 15:45:00")].iloc[0]
    assert row_1400["minutes_to_close"] == pytest.approx(120.0)
    assert row_1545["minutes_to_close"] == pytest.approx(15.0)


def test_today_return_so_far_uses_session_open_close():
    day = _dev_day("2024-03-01", [
        ("13:00:00", np.nan, True, 100.0),   # session open reference
        ("14:00:00", 500.0, False, 102.0),
    ])
    feat = add_dynamics_features(day)
    row = feat[feat["time"] == pd.Timestamp("2024-03-01 14:00:00")].iloc[0]
    assert row["today_return_so_far"] == pytest.approx(0.02)
