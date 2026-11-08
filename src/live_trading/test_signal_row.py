from datetime import datetime

import pandas as pd
import pytest

from live_trading.signal_row import assemble_live_signal_row, SIGNAL_HORIZONS


def test_row_carries_basic_fields():
    row = assemble_live_signal_row(
        now=datetime(2026, 7, 21, 14, 30), close=550.0, minutes_to_close=90.0,
        density_pred={}, touch_pred={}, agg_imbalance=1.5e9, imb_z=1.2,
        imb_velocity=1e7, imb_acceleration=-1e6,
    )
    assert row["close"] == row["high"] == row["low"] == 550.0
    assert row["aggregate_dollar_imbalance"] == 1.5e9
    assert row["imbalance_zscore_intraday"] == pytest.approx(1.2)
    assert row["minutes_to_close"] == 90.0


def test_row_carries_imbalance_zscore_tod():
    row = assemble_live_signal_row(
        now=datetime(2026, 7, 21, 14, 30), close=550.0, minutes_to_close=90.0,
        density_pred={}, touch_pred={}, agg_imbalance=1.5e9, imb_z=1.2, imb_z_tod=-0.8,
        imb_velocity=1e7, imb_acceleration=-1e6,
    )
    assert row["imbalance_zscore_tod"] == pytest.approx(-0.8)


def test_row_imbalance_zscore_tod_defaults_to_nan():
    # A caller not yet passing imb_z_tod (e.g. an older code path) must
    # degrade to NaN, not raise -- matches every other optional-signal
    # convention in assemble_live_signal_row.
    row = assemble_live_signal_row(
        now=datetime(2026, 7, 21, 14, 30), close=550.0, minutes_to_close=90.0,
        density_pred={}, touch_pred={}, agg_imbalance=1.5e9, imb_z=1.2,
        imb_velocity=1e7, imb_acceleration=-1e6,
    )
    assert pd.isna(row["imbalance_zscore_tod"])


def test_missing_horizon_becomes_nan_matching_engine_censoring():
    # Only 5min present in the prediction dicts -- mirrors both frozen
    # engines' own predict() output when later horizons' exits fall past
    # 16:00 (they're just absent, not present-with-NaN).
    density_pred = {"5min": {"P_tail": 0.3, "P_upside": 0.15, "P_downside": 0.15}}
    touch_pred = {"5min": {"P_touch": 0.4}}
    row = assemble_live_signal_row(
        now=datetime(2026, 7, 21, 15, 58), close=550.0, minutes_to_close=2.0,
        density_pred=density_pred, touch_pred=touch_pred, agg_imbalance=0.0, imb_z=float("nan"),
        imb_velocity=float("nan"), imb_acceleration=float("nan"),
    )
    assert row["p_tail_5min"] == pytest.approx(0.3)
    assert row["p_touch_5min"] == pytest.approx(0.4)
    for h in [h for h in SIGNAL_HORIZONS if h != "5min"]:
        assert pd.isna(row[f"p_tail_{h}"])
        assert pd.isna(row[f"p_touch_{h}"])


@pytest.mark.parametrize("imb,vel,acc,expect_reverting,expect_accelerating", [
    (1e9, -1e8, -1e6, True, False),   # positive imbalance, vel opposite sign -> reverting;
                                       # acc same sign as imb*vel<0 pairing -> not accelerating
    (1e9, 1e8, 1e6, False, True),     # positive imbalance, vel same sign -> not reverting;
                                       # acc same sign as imb -> accelerating
    (-1e9, 1e8, 1e6, True, False),    # negative imbalance, vel opposite sign -> reverting;
                                       # acc opposite sign to imb -> not accelerating
])
def test_reverting_accelerating_match_documented_formula(imb, vel, acc, expect_reverting, expect_accelerating):
    # Matches transition_state_features.py / backtest_signals.py's formula
    # exactly: reverting = (imb * vel) < 0, accelerating = (imb * acc) > 0.
    row = assemble_live_signal_row(
        now=datetime(2026, 7, 21, 14, 30), close=550.0, minutes_to_close=90.0,
        density_pred={}, touch_pred={}, agg_imbalance=imb, imb_z=0.0,
        imb_velocity=vel, imb_acceleration=acc,
    )
    assert bool(row["imbalance_reverting"]) is expect_reverting
    assert bool(row["imbalance_accelerating"]) is expect_accelerating


def test_row_has_every_column_backtest_engine_functions_read():
    """Direct contract check against the actual reused decision functions
    (research repo's `backtest_engine.py`) -- rather than reproducing that
    repo's full historical feature-grid pipeline just to diff against it
    (a much heavier, more brittle test for the same guarantee), this
    proves the live row satisfies exactly the attribute contract
    `_passes_filters`/`_entry_direction`/`_size_position`/`_open_trade`/
    `_check_exit` actually read via `getattr`/attribute access."""
    import backtest_engine as be
    from strategy_components import StrategyConfig

    row = assemble_live_signal_row(
        now=datetime(2026, 7, 21, 14, 30), close=550.0, minutes_to_close=90.0,
        density_pred={"5min": {"P_tail": 0.3, "P_upside": 0.15, "P_downside": 0.15}},
        touch_pred={"5min": {"P_touch": 0.6}},
        agg_imbalance=1e9, imb_z=2.0, imb_velocity=1e7, imb_acceleration=1e6,
    )
    cfg = StrategyConfig(prob_metric="p_touch", entry_horizon="5min", prob_threshold=0.5,
                         direction_mode="both")
    t_start, t_end = be._parse_time(cfg.time_start), be._parse_time(cfg.time_end)
    # None of these should raise AttributeError -- that's the contract being tested.
    assert be._passes_filters(row, cfg, t_start, t_end) is True
    direction = be._entry_direction(row, cfg)
    assert direction == 1
    frac = be._size_position(row, cfg)
    assert frac > 0
    pos = be._open_trade(row, direction, frac, cfg)
    exit_info = be._check_exit(pos, row, be._parse_time(cfg.eod_exit_time))
    assert exit_info is None  # nothing should trigger an exit on the entry bar itself


def test_row_satisfies_zscore_sign_direction_and_tod_zscore_source_contract():
    # Same contract check as above, but for the newer direction_source=
    # "zscore_sign" + zscore_source="time_of_day_normalized" combination
    # -- the exact config a deployed strategy would use to test the
    # gap found in live paper trading (raw imbalance sign vs. z-score
    # sign disagreeing). imbalance_zscore_tod is negative here while the
    # raw imbalance level is positive -- direction must come out SHORT.
    import backtest_engine as be
    from strategy_components import StrategyConfig

    row = assemble_live_signal_row(
        now=datetime(2026, 7, 21, 14, 30), close=550.0, minutes_to_close=90.0,
        density_pred={"5min": {"P_tail": 0.3, "P_upside": 0.15, "P_downside": 0.15}},
        touch_pred={"5min": {"P_touch": 0.6}},
        agg_imbalance=5e8, imb_z=2.0, imb_z_tod=-1.5, imb_velocity=1e7, imb_acceleration=1e6,
    )
    cfg = StrategyConfig(prob_metric="p_touch", entry_horizon="5min", prob_threshold=0.5,
                         direction_mode="both", direction_source="zscore_sign",
                         zscore_source="time_of_day_normalized", min_abs_zscore=1.0)
    t_start, t_end = be._parse_time(cfg.time_start), be._parse_time(cfg.time_end)
    assert be._passes_filters(row, cfg, t_start, t_end) is True
    assert be._entry_direction(row, cfg) == -1  # imbalance_zscore_tod is negative -> short
