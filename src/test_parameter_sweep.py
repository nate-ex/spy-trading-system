"""Tests for parameter_sweep.py -- synthetic signals, no real-data
dependency."""

from __future__ import annotations

import pandas as pd
import pytest

from parameter_sweep import grid_search, random_search
from strategy_components import StrategyConfig

BASE_COLS = ["time", "date", "open", "high", "low", "close", "minutes_to_close",
            "aggregate_dollar_imbalance", "imbalance_zscore_intraday", "imbalance_velocity",
            "imbalance_acceleration", "imbalance_stale", "imbalance_reverting", "imbalance_accelerating",
            "p_touch_5min", "p_tail_5min", "p_upside_5min", "p_downside_5min"]


def _bar(t, close, p_touch, imb=1e9):
    return {
        "time": pd.Timestamp(t), "date": pd.Timestamp(t).normalize(),
        "open": close, "high": close, "low": close, "close": close, "minutes_to_close": 60.0,
        "aggregate_dollar_imbalance": imb, "imbalance_zscore_intraday": 1.0, "imbalance_velocity": 0.0,
        "imbalance_acceleration": 0.0, "imbalance_stale": False,
        "imbalance_reverting": False, "imbalance_accelerating": False,
        "p_touch_5min": p_touch, "p_tail_5min": p_touch, "p_upside_5min": p_touch / 2, "p_downside_5min": p_touch / 2,
    }


@pytest.fixture
def signals():
    rows = [
        _bar("2024-01-02 14:00", 100.00, 0.9),
        _bar("2024-01-02 14:01", 100.50, 0.1),
        _bar("2024-01-02 14:02", 100.20, 0.1),
        _bar("2024-01-03 14:00", 100.00, 0.9),
        _bar("2024-01-03 14:01", 99.50, 0.1),
        _bar("2024-01-03 14:02", 99.80, 0.1),
    ]
    return pd.DataFrame(rows, columns=BASE_COLS)


def test_grid_search_covers_full_cartesian_product(signals):
    base = StrategyConfig(prob_threshold=0.5)
    grid = {"prob_threshold": [0.5, 0.7], "direction_mode": ["long_only", "short_only"]}
    out = grid_search(signals, base, grid)
    assert len(out) == 4  # 2 x 2
    assert set(out["n_variants_tested"]) == {4}
    assert set(out["prob_threshold"]) == {0.5, 0.7}
    assert set(out["direction_mode"]) == {"long_only", "short_only"}


def test_grid_search_includes_metrics_columns(signals):
    base = StrategyConfig()
    out = grid_search(signals, base, {"prob_threshold": [0.5]})
    for col in ["total_pnl", "win_rate", "n_trades", "sharpe", "max_drawdown"]:
        assert col in out.columns


def test_random_search_respects_n_iter_cap(signals):
    base = StrategyConfig()
    space = {"prob_threshold": [0.4, 0.5, 0.6], "fixed_fraction": [0.5, 1.0]}
    out = random_search(signals, base, space, n_iter=4, random_state=1)
    assert len(out) == 4


def test_random_search_deterministic_with_seed(signals):
    base = StrategyConfig()
    space = {"prob_threshold": [0.4, 0.5, 0.6, 0.7], "fixed_fraction": [0.5, 1.0, 1.5]}
    out1 = random_search(signals, base, space, n_iter=5, random_state=42)
    out2 = random_search(signals, base, space, n_iter=5, random_state=42)
    assert list(out1["prob_threshold"]) == list(out2["prob_threshold"])
    assert list(out1["fixed_fraction"]) == list(out2["fixed_fraction"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
