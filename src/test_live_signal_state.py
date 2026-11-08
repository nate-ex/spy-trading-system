"""Tests for the new time-of-day-normalized live z-score
(`live_imbalance_zscore_tod`/`_nearest_tod_std`). Synthetic data only, no
dependency on the exported baseline file on disk."""

from __future__ import annotations

from datetime import time

import pandas as pd
import pytest

from live_signal_state import live_imbalance_zscore_tod, _nearest_tod_std


def test_nearest_tod_std_exact_match():
    baseline = {"14:00": 100.0, "14:05": 200.0, "14:10": 300.0}
    assert _nearest_tod_std(baseline, time(14, 5)) == 200.0


def test_nearest_tod_std_between_grid_points_uses_prior():
    baseline = {"14:00": 100.0, "14:05": 200.0, "14:10": 300.0}
    assert _nearest_tod_std(baseline, time(14, 7)) == 200.0


def test_nearest_tod_std_before_all_entries_falls_back_to_earliest():
    baseline = {"14:00": 100.0, "14:05": 200.0}
    assert _nearest_tod_std(baseline, time(13, 30)) == 100.0


def test_nearest_tod_std_empty_baseline_returns_none():
    assert _nearest_tod_std({}, time(14, 0)) is None


def test_live_zscore_tod_hand_computed():
    prior_hist = pd.DataFrame({"aggregate_dollar_imbalance": [10.0, 20.0, 30.0]})
    baseline = {"14:00": 5.0}
    z = live_imbalance_zscore_tod(50.0, prior_hist, time(14, 0), baseline=baseline)
    # prior mean = 20.0, std from baseline = 5.0
    assert z == pytest.approx((50.0 - 20.0) / 5.0)


def test_live_zscore_tod_nan_before_min_periods():
    prior_hist = pd.DataFrame({"aggregate_dollar_imbalance": [10.0, 20.0]})
    baseline = {"14:00": 5.0}
    z = live_imbalance_zscore_tod(50.0, prior_hist, time(14, 0), min_periods=3, baseline=baseline)
    assert pd.isna(z)


def test_live_zscore_tod_nan_when_baseline_std_missing():
    prior_hist = pd.DataFrame({"aggregate_dollar_imbalance": [10.0, 20.0, 30.0]})
    z = live_imbalance_zscore_tod(50.0, prior_hist, time(14, 0), baseline={})
    assert pd.isna(z)


def test_live_zscore_tod_nan_when_baseline_std_zero():
    prior_hist = pd.DataFrame({"aggregate_dollar_imbalance": [10.0, 20.0, 30.0]})
    z = live_imbalance_zscore_tod(50.0, prior_hist, time(14, 0), baseline={"14:00": 0.0})
    assert pd.isna(z)


def test_live_zscore_tod_differs_from_intraday_at_same_reading():
    # The whole point of this feature: with an identical current reading
    # and prior history, the two z-scores can disagree because their
    # denominators come from different populations (same-day-only vs.
    # cross-day time-of-day-matched).
    from live_signal_state import live_imbalance_zscore_intraday

    prior_hist = pd.DataFrame({"aggregate_dollar_imbalance": [10.0, 10.0, 10.0]})  # same-day std = 0 -> NaN
    z_intraday = live_imbalance_zscore_intraday(50.0, prior_hist)
    z_tod = live_imbalance_zscore_tod(50.0, prior_hist, time(14, 0), baseline={"14:00": 20.0})
    assert pd.isna(z_intraday)
    assert z_tod == pytest.approx((50.0 - 10.0) / 20.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
