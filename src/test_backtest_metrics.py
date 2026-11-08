"""Tests for backtest_metrics.py's formulas -- hand-verified on small
synthetic trade/equity data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

import backtest_metrics as bm


@dataclass
class _FakeResult:
    trades: pd.DataFrame
    equity: pd.DataFrame


def _trades(rows):
    return pd.DataFrame(rows)


def test_max_drawdown_on_a_known_equity_path():
    # equity: 0 -> 0.05 -> 0.02 -> 0.08 -> 0.01
    # running max: 0, .05, .05, .08, .08 -> drawdowns: 0,0,-.03,0,-.07
    eq = pd.Series([0.0, 0.05, 0.02, 0.08, 0.01])
    assert bm.max_drawdown(eq) == pytest.approx(-0.07)


def test_profit_factor_known_ratio():
    trades = _trades([{"pnl": 0.02}, {"pnl": 0.03}, {"pnl": -0.01}, {"pnl": -0.02}])
    # wins = 0.05, losses = 0.03 -> profit factor = 0.05/0.03
    assert bm.profit_factor(trades) == pytest.approx(0.05 / 0.03)


def test_profit_factor_no_losses_is_infinite():
    trades = _trades([{"pnl": 0.02}, {"pnl": 0.01}])
    assert bm.profit_factor(trades) == float("inf")


def test_sharpe_zero_when_flat():
    daily = pd.Series([0.01, 0.01, 0.01])  # zero variance
    assert bm.sharpe_ratio(daily) == 0.0


def test_sharpe_matches_hand_computation():
    daily = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01])
    expected = daily.mean() / daily.std(ddof=1) * np.sqrt(252)
    assert bm.sharpe_ratio(daily) == pytest.approx(expected)


def test_sortino_only_penalizes_downside():
    daily = pd.Series([0.01, 0.05, 0.10, -0.01, -0.02])
    downside = daily[daily < 0]
    expected = daily.mean() / downside.std(ddof=1) * np.sqrt(252)
    assert bm.sortino_ratio(daily) == pytest.approx(expected)


def test_pnl_by_time_of_day_buckets_correctly():
    trades = _trades([
        {"entry_time": pd.Timestamp("2024-01-02 14:03"), "pnl": 0.01},
        {"entry_time": pd.Timestamp("2024-01-02 14:07"), "pnl": 0.02},
        {"entry_time": pd.Timestamp("2024-01-02 14:20"), "pnl": -0.01},
    ])
    out = bm.pnl_by_time_of_day(trades, bucket_minutes=15).set_index("bucket")
    assert out.loc["14:00", "total_pnl"] == pytest.approx(0.03)
    assert out.loc["14:00", "n_trades"] == 2
    assert out.loc["14:15", "total_pnl"] == pytest.approx(-0.01)


def test_compute_metrics_empty_trades_returns_zeroed_dict():
    result = _FakeResult(trades=pd.DataFrame(columns=["pnl", "exit_time", "holding_minutes"]),
                         equity=pd.DataFrame({"time": [], "equity": [], "exposure": []}))
    m = bm.compute_metrics(result)
    assert m["n_trades"] == 0
    assert m["total_pnl"] == 0.0


def test_compute_metrics_basic_fields():
    trades = _trades([
        {"entry_time": pd.Timestamp("2024-01-02 14:00"), "exit_time": pd.Timestamp("2024-01-02 14:05"),
         "pnl": 0.02, "holding_minutes": 5.0},
        {"entry_time": pd.Timestamp("2024-01-03 14:00"), "exit_time": pd.Timestamp("2024-01-03 14:10"),
         "pnl": -0.01, "holding_minutes": 10.0},
    ])
    equity = pd.DataFrame({"time": pd.to_datetime(["2024-01-02 14:05", "2024-01-03 14:10"]),
                          "equity": [0.02, 0.01], "exposure": [1.0, 1.0]})
    result = _FakeResult(trades=trades, equity=equity)
    m = bm.compute_metrics(result)
    assert m["n_trades"] == 2
    assert m["total_pnl"] == pytest.approx(0.01)
    assert m["win_rate"] == pytest.approx(0.5)
    assert m["avg_holding_minutes"] == pytest.approx(7.5)
    assert m["avg_exposure"] == pytest.approx(1.0)


def _signals_day(date_str: str, times_and_closes: list) -> pd.DataFrame:
    rows = [{"date": pd.Timestamp(date_str), "time": pd.Timestamp(f"{date_str} {t}"), "close": c}
           for t, c in times_and_closes]
    return pd.DataFrame(rows)


def test_buy_hold_benchmark_hand_verified_two_days():
    # Day 1: 14:00 close=500, 16:00 close=505 -> +1.0%
    # Day 2: 14:00 close=500, 16:00 close=495 -> -1.0%
    day1 = _signals_day("2024-01-02", [("14:00", 500.0), ("15:00", 502.0), ("16:00", 505.0)])
    day2 = _signals_day("2024-01-03", [("14:00", 500.0), ("15:00", 498.0), ("16:00", 495.0)])
    signals = pd.concat([day1, day2], ignore_index=True)

    bh = bm.buy_hold_benchmark(signals, pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-01-03").date())
    assert bh["n_trades"] == 2
    assert bh["total_pnl"] == pytest.approx(0.01 - 0.01, abs=1e-9)
    assert bh["avg_pnl_per_trade"] == pytest.approx(0.0, abs=1e-9)
    assert bh["win_rate"] == pytest.approx(0.5)
    assert bh["avg_exposure"] == pytest.approx(1.0)
    assert bh["avg_holding_minutes"] == pytest.approx(120.0)  # 14:00 -> 16:00


def test_buy_hold_benchmark_uses_first_and_last_bar_in_window_not_whole_day():
    # A bar at 13:00 (before time_start) and one at 16:30 (after time_end)
    # must be ignored -- only 14:00->16:00 counts.
    day = _signals_day("2024-01-02", [
        ("13:00", 400.0), ("14:00", 500.0), ("15:30", 550.0), ("16:00", 510.0), ("16:30", 900.0),
    ])
    bh = bm.buy_hold_benchmark(day, pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-01-02").date())
    assert bh["total_pnl"] == pytest.approx(510.0 / 500.0 - 1, abs=1e-9)


def test_buy_hold_benchmark_respects_date_range_filter():
    day1 = _signals_day("2024-01-02", [("14:00", 500.0), ("16:00", 505.0)])
    day2 = _signals_day("2024-01-03", [("14:00", 500.0), ("16:00", 495.0)])
    signals = pd.concat([day1, day2], ignore_index=True)
    bh = bm.buy_hold_benchmark(signals, pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-01-02").date())
    assert bh["n_trades"] == 1
    assert bh["total_pnl"] == pytest.approx(0.01, abs=1e-9)


def test_buy_hold_benchmark_empty_range_returns_zeroed_dict():
    day = _signals_day("2024-01-02", [("14:00", 500.0), ("16:00", 505.0)])
    bh = bm.buy_hold_benchmark(day, pd.Timestamp("2020-01-01").date(), pd.Timestamp("2020-01-02").date())
    assert bh["n_trades"] == 0
    assert bh["total_pnl"] == 0.0
    assert bh["avg_exposure"] == 1.0  # still fully-invested BY CONSTRUCTION even with zero days


def test_buy_hold_benchmark_custom_window():
    day = _signals_day("2024-01-02", [("14:00", 500.0), ("14:30", 502.0), ("15:00", 510.0), ("16:00", 520.0)])
    bh = bm.buy_hold_benchmark(day, pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-01-02").date(),
                               time_start="14:00", time_end="15:00")
    assert bh["total_pnl"] == pytest.approx(510.0 / 500.0 - 1, abs=1e-9)
    assert bh["avg_holding_minutes"] == pytest.approx(60.0)


def test_buy_hold_benchmark_output_keys_match_compute_metrics_keys():
    # Must slot into the same Compare-tab table (dashboard_trading.METRIC_COLS)
    # as a strategy's compute_metrics() output, no missing columns.
    day = _signals_day("2024-01-02", [("14:00", 500.0), ("16:00", 505.0)])
    bh = bm.buy_hold_benchmark(day, pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-01-02").date())
    strategy_keys = {"total_pnl", "avg_pnl_per_trade", "win_rate", "n_trades", "max_drawdown",
                     "sharpe", "sortino", "profit_factor", "avg_holding_minutes", "avg_exposure"}
    assert strategy_keys <= set(bh.keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
