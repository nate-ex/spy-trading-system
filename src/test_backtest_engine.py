"""Correctness tests for backtest_engine.py's non-obvious simulation logic
-- hand-verified expected trades on small synthetic bar sequences, per
CLAUDE.md's testing convention. No real-data dependency."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest_engine import run_backtest
from strategy_components import StrategyConfig

BASE_COLS = ["time", "date", "open", "high", "low", "close", "minutes_to_close",
            "aggregate_dollar_imbalance", "imbalance_zscore_intraday", "imbalance_zscore_tod",
            "imbalance_velocity", "imbalance_acceleration", "imbalance_stale", "imbalance_reverting",
            "imbalance_accelerating", "p_touch_5min", "p_tail_5min", "p_upside_5min", "p_downside_5min"]


def _bar(t, close, high=None, low=None, p_touch=0.0, imb=1e9, z=1.0, z_tod=None,
        reverting=False, accelerating=False):
    high = close if high is None else high
    low = close if low is None else low
    z_tod = z if z_tod is None else z_tod
    return {
        "time": pd.Timestamp(t), "date": pd.Timestamp(t).normalize(),
        "open": close, "high": high, "low": low, "close": close, "minutes_to_close": 60.0,
        "aggregate_dollar_imbalance": imb, "imbalance_zscore_intraday": z, "imbalance_zscore_tod": z_tod,
        "imbalance_velocity": 0.0,
        "imbalance_acceleration": 0.0, "imbalance_stale": False,
        "imbalance_reverting": reverting, "imbalance_accelerating": accelerating,
        "p_touch_5min": p_touch, "p_tail_5min": p_touch, "p_upside_5min": p_touch / 2, "p_downside_5min": p_touch / 2,
    }


def _df(rows):
    return pd.DataFrame(rows, columns=BASE_COLS)


def _cfg(**kw):
    base = dict(entry_horizon="5min", prob_metric="p_touch", prob_threshold=0.6,
               direction_mode="both", sizing_mode="fixed", fixed_fraction=1.0,
               time_start="14:00", time_end="16:00", eod_exit_time="16:00")
    base.update(kw)
    return StrategyConfig(**base)


def test_profit_target_exit_skips_the_entry_bars_own_high():
    # Entry bar's own high (200) would hit any realistic profit target --
    # if the engine incorrectly checked exits on the entry bar itself,
    # this trade would close instantly at an unrealistic price. It must
    # not exit until a LATER bar's high crosses the target.
    rows = [
        _bar("2024-01-02 14:00", close=100.00, high=200.00, p_touch=0.7, imb=1e9),  # entry (own high ignored)
        _bar("2024-01-02 14:01", close=100.10, high=100.15, low=100.05, p_touch=0.1),  # target not yet hit
        _bar("2024-01-02 14:02", close=100.60, high=100.60, low=100.55, p_touch=0.1),  # target hit here
        _bar("2024-01-02 14:03", close=100.60, p_touch=0.1),
    ]
    cfg = _cfg(profit_target_bps=50, stop_loss_bps=30)
    result = run_backtest(_df(rows), cfg)
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["entry_time"] == pd.Timestamp("2024-01-02 14:00")
    assert trade["exit_time"] == pd.Timestamp("2024-01-02 14:02")
    assert trade["exit_reason"] == "profit_target"
    assert trade["exit_price"] == pytest.approx(100.50)  # entry * (1 + 50bps)
    assert trade["pnl"] == pytest.approx(0.005)


def test_stop_loss_wins_priority_when_both_touched_same_bar():
    rows = [
        _bar("2024-01-02 14:00", close=100.00, p_touch=0.7, imb=1e9),  # long entry
        _bar("2024-01-02 14:01", close=100.00, high=100.60, low=99.60, p_touch=0.1),  # both target & stop touched
    ]
    cfg = _cfg(profit_target_bps=50, stop_loss_bps=30)
    result = run_backtest(_df(rows), cfg)
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == "stop_loss"
    assert result.trades.iloc[0]["exit_price"] == pytest.approx(99.70)


def test_eod_force_close_when_no_other_exit_fires():
    rows = [
        _bar("2024-01-02 14:00", close=100.00, p_touch=0.7, imb=1e9),
        _bar("2024-01-02 14:01", close=100.05, p_touch=0.1),
        _bar("2024-01-02 14:02", close=100.02, p_touch=0.1),
    ]
    cfg = _cfg()  # no profit target / stop loss / holding period
    result = run_backtest(_df(rows), cfg)
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "eod"
    assert trade["exit_time"] == pd.Timestamp("2024-01-02 14:02")
    assert trade["exit_price"] == pytest.approx(100.02)


def test_holding_period_exit():
    rows = [
        _bar("2024-01-02 14:00", close=100.00, p_touch=0.7, imb=1e9),
        _bar("2024-01-02 14:01", close=100.01, p_touch=0.1),
        _bar("2024-01-02 14:02", close=100.02, p_touch=0.1),
        _bar("2024-01-02 14:05", close=100.05, p_touch=0.1),
    ]
    cfg = _cfg(holding_period_min=2)
    result = run_backtest(_df(rows), cfg)
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == "holding_period"
    assert result.trades.iloc[0]["exit_time"] == pd.Timestamp("2024-01-02 14:02")


def test_max_total_exposure_blocks_pyramiding_beyond_the_cap():
    rows = [
        _bar("2024-01-02 14:00", close=100.00, p_touch=0.9, imb=1e9),
        _bar("2024-01-02 14:01", close=100.00, p_touch=0.9, imb=1e9),  # would open a 2nd position
        _bar("2024-01-02 14:02", close=100.00, p_touch=0.1),
    ]
    cfg = _cfg(fixed_fraction=1.0, max_total_exposure=1.5)
    result = run_backtest(_df(rows), cfg)
    # first position uses 1.0 of the 1.5 cap; the second (1.0 more) would
    # breach it and must be skipped -- exactly one trade opened.
    assert len(result.trades) == 1


def test_direction_both_mode_takes_direction_from_imbalance_sign():
    rows = [
        _bar("2024-01-02 14:00", close=100.00, p_touch=0.9, imb=-5e8),  # negative imbalance -> short
        _bar("2024-01-02 14:01", close=99.00, p_touch=0.1),
    ]
    cfg = _cfg(direction_mode="both")
    result = run_backtest(_df(rows), cfg)
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["direction"] == -1
    # short from 100 -> 99 is a +1% gain for the short.
    assert result.trades.iloc[0]["pnl"] == pytest.approx(0.01)


def test_long_only_mode_ignores_imbalance_sign():
    rows = [
        _bar("2024-01-02 14:00", close=100.00, p_touch=0.9, imb=-5e8),  # negative imbalance
        _bar("2024-01-02 14:01", close=101.00, p_touch=0.1),
    ]
    cfg = _cfg(direction_mode="long_only")
    result = run_backtest(_df(rows), cfg)
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["direction"] == 1


def test_imbalance_direction_filter_blocks_disagreeing_entries():
    rows = [
        _bar("2024-01-02 14:00", close=100.00, p_touch=0.9, imb=-5e8),  # negative imbalance
        _bar("2024-01-02 14:01", close=100.00, p_touch=0.1),
    ]
    cfg = _cfg(imbalance_direction="positive")  # requires positive imbalance -> should block
    result = run_backtest(_df(rows), cfg)
    assert len(result.trades) == 0


def test_min_abs_zscore_filter():
    rows = [
        _bar("2024-01-02 14:00", close=100.00, p_touch=0.9, imb=1e9, z=0.5),
        _bar("2024-01-02 14:01", close=100.00, p_touch=0.1),
    ]
    cfg = _cfg(min_abs_zscore=1.0)
    result = run_backtest(_df(rows), cfg)
    assert len(result.trades) == 0  # |z|=0.5 < required 1.0


def test_confidence_weighted_sizing_formula():
    rows = [
        _bar("2024-01-02 14:00", close=100.00, p_touch=0.8, imb=1e9),
        _bar("2024-01-02 14:01", close=100.00, p_touch=0.1),
    ]
    cfg = _cfg(sizing_mode="confidence_weighted", confidence_scale=1.0)
    result = run_backtest(_df(rows), cfg)
    assert len(result.trades) == 1
    # confidence = (0.8 - 0.5) * 2 = 0.6
    assert result.trades.iloc[0]["fraction"] == pytest.approx(0.6)


def test_prob_below_threshold_never_enters():
    rows = [
        _bar("2024-01-02 14:00", close=100.00, p_touch=0.4, imb=1e9),
        _bar("2024-01-02 14:01", close=105.00, p_touch=0.1),
    ]
    cfg = _cfg(prob_threshold=0.6)
    result = run_backtest(_df(rows), cfg)
    assert len(result.trades) == 0


def test_cost_bps_reduces_pnl():
    rows = [
        _bar("2024-01-02 14:00", close=100.00, p_touch=0.9, imb=1e9),
        _bar("2024-01-02 14:01", close=101.00, p_touch=0.1),
    ]
    cfg = _cfg(cost_bps=20.0)  # 20bps round-trip
    result = run_backtest(_df(rows), cfg)
    assert result.trades.iloc[0]["pnl"] == pytest.approx(0.01 - 0.0020)


def test_equity_curve_tracks_realized_and_unrealized_pnl():
    rows = [
        _bar("2024-01-02 14:00", close=100.00, p_touch=0.9, imb=1e9),
        _bar("2024-01-02 14:01", close=101.00, p_touch=0.1),  # still open, +1% unrealized
        _bar("2024-01-02 14:02", close=101.00, p_touch=0.1),
    ]
    cfg = _cfg()  # no exit rule -> holds to EOD (last bar)
    result = run_backtest(_df(rows), cfg)
    eq = result.equity.set_index("time")["equity"]
    assert eq.loc[pd.Timestamp("2024-01-02 14:01")] == pytest.approx(0.01)  # unrealized only
    assert eq.loc[pd.Timestamp("2024-01-02 14:02")] == pytest.approx(0.01)  # realized at EOD close


def test_entry_horizon_20min_and_30min_columns_drive_entries_too():
    # backtest_engine.py has no horizon-specific logic -- entry_horizon is
    # just a string interpolated into f"{prob_metric}_{entry_horizon}"
    # (_passes_filters/_size_position). This is a smoke test confirming
    # that generic path actually works for the newly-added 20min/30min
    # horizons, not just the 5min the other tests in this file exercise.
    cols = BASE_COLS + ["p_touch_20min", "p_tail_20min", "p_upside_20min", "p_downside_20min",
                        "p_touch_30min", "p_tail_30min", "p_upside_30min", "p_downside_30min"]
    row = _bar("2024-01-02 14:00", close=100.00, p_touch=0.0, imb=1e9)  # 5min columns unused here
    for h in ("20min", "30min"):
        row[f"p_touch_{h}"] = 0.9; row[f"p_tail_{h}"] = 0.9
        row[f"p_upside_{h}"] = 0.45; row[f"p_downside_{h}"] = 0.45
    df = pd.DataFrame([row], columns=cols)

    for h in ("20min", "30min"):
        cfg = _cfg(entry_horizon=h)
        result = run_backtest(df, cfg)
        assert len(result.trades) == 1, f"expected one entry for entry_horizon={h}"


def test_direction_source_zscore_sign_can_disagree_with_raw_imbalance():
    # The exact gap that motivated this feature: a large-but-typical-for-
    # today imbalance (raw sign positive) can have a z-score near/below
    # zero. direction_source="zscore_sign" should go SHORT here (z<0)
    # even though raw_imbalance would have gone LONG (imb>0).
    rows = [
        _bar("2024-01-02 14:00", close=100.00, p_touch=0.9, imb=5e8, z=-0.5),
        _bar("2024-01-02 14:01", close=100.50, p_touch=0.1),
    ]
    cfg_raw = _cfg(direction_source="raw_imbalance")
    cfg_zsign = _cfg(direction_source="zscore_sign")

    result_raw = run_backtest(_df(rows), cfg_raw)
    result_zsign = run_backtest(_df(rows), cfg_zsign)

    assert len(result_raw.trades) == 1
    assert result_raw.trades.iloc[0]["direction"] == 1  # raw imbalance positive -> long
    assert len(result_zsign.trades) == 1
    assert result_zsign.trades.iloc[0]["direction"] == -1  # z negative -> short


def test_direction_source_zscore_sign_flat_when_zscore_nan():
    rows = [
        _bar("2024-01-02 14:00", close=100.00, p_touch=0.9, imb=5e8, z=float("nan")),
    ]
    cfg = _cfg(direction_source="zscore_sign")
    result = run_backtest(_df(rows), cfg)
    assert len(result.trades) == 0


def test_zscore_source_time_of_day_normalized_reads_the_tod_column():
    # imbalance_zscore_intraday and imbalance_zscore_tod disagree in sign
    # here -- zscore_source picks which one both min_abs_zscore and
    # direction_source="zscore_sign" actually read.
    rows = [
        _bar("2024-01-02 14:00", close=100.00, p_touch=0.9, imb=5e8, z=-1.0, z_tod=1.0),
        _bar("2024-01-02 14:01", close=100.50, p_touch=0.1),
    ]
    cfg = _cfg(direction_source="zscore_sign", zscore_source="time_of_day_normalized",
              min_abs_zscore=0.5)
    result = run_backtest(_df(rows), cfg)
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["direction"] == 1  # imbalance_zscore_tod is positive


def test_min_abs_zscore_gate_respects_zscore_source():
    # |imbalance_zscore_tod| (0.2) is below the 0.5 gate even though
    # |imbalance_zscore_intraday| (2.0) clears it -- with
    # zscore_source="time_of_day_normalized" the trade must be filtered.
    rows = [
        _bar("2024-01-02 14:00", close=100.00, p_touch=0.9, imb=5e8, z=2.0, z_tod=0.2),
    ]
    cfg = _cfg(min_abs_zscore=0.5, zscore_source="time_of_day_normalized")
    result = run_backtest(_df(rows), cfg)
    assert len(result.trades) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
