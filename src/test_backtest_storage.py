"""Tests for backtest_storage.py -- config save/load (existing behavior,
locked in), and the comparison-table/last-sweep autosave/autoload added to
survive LIVE mode's auto-refresh (a full browser reload -> fresh
st.session_state every poll) and next-day relaunches."""
from __future__ import annotations

import pandas as pd
import pytest

import backtest_storage as bs
from strategy_components import StrategyConfig


@pytest.fixture(autouse=True)
def _isolated_backtest_dir(tmp_path, monkeypatch):
    """Redirect every module-level path constant into a fresh tmp_path per
    test, so tests never touch (or depend on) the real data/processed/backtests/."""
    backtest_dir = tmp_path / "backtests"
    monkeypatch.setattr(bs, "BACKTEST_DIR", backtest_dir)
    monkeypatch.setattr(bs, "CONFIGS_DIR", backtest_dir / "configs")
    monkeypatch.setattr(bs, "RESULTS_DIR", backtest_dir / "results")
    monkeypatch.setattr(bs, "COMPARISON_PATH", backtest_dir / "comparison.json")
    return backtest_dir


def test_list_saved_configs_empty_when_dir_missing():
    assert bs.list_saved_configs() == []


def test_save_load_config_roundtrip():
    cfg = StrategyConfig(name="my strategy", prob_threshold=0.7)
    bs.save_config(cfg)
    loaded = bs.load_config("my_strategy")  # spaces sanitized to underscores
    assert loaded.name == "my strategy" and loaded.prob_threshold == 0.7
    assert bs.list_saved_configs() == ["my_strategy"]


def test_delete_config_removes_file():
    cfg = StrategyConfig(name="temp")
    bs.save_config(cfg)
    assert "temp" in bs.list_saved_configs()
    bs.delete_config("temp")
    assert "temp" not in bs.list_saved_configs()


def test_delete_config_missing_file_is_a_noop():
    bs.delete_config("never-existed")  # must not raise


def test_save_load_sweep_results_roundtrip():
    df = pd.DataFrame({"name": ["a", "b"], "sharpe": [1.2, 0.8]})
    bs.save_sweep_results(df, "my sweep")
    loaded = bs.load_sweep_results("my_sweep")
    pd.testing.assert_frame_equal(loaded, df)
    assert bs.list_saved_sweeps() == ["my_sweep"]


def test_list_named_sweeps_excludes_autosave_slot():
    bs.save_sweep_results(pd.DataFrame({"x": [1]}), "real_sweep")
    bs.save_last_sweep(pd.DataFrame({"x": [2]}))
    assert bs.list_saved_sweeps() == sorted(["real_sweep", bs.LAST_SWEEP_NAME])
    assert bs.list_named_sweeps() == ["real_sweep"]


def test_save_load_last_sweep_roundtrip():
    df = pd.DataFrame({"name": ["a"], "sharpe": [1.5]})
    bs.save_last_sweep(df)
    loaded = bs.load_last_sweep()
    pd.testing.assert_frame_equal(loaded, df)


def test_load_last_sweep_none_when_never_saved():
    assert bs.load_last_sweep() is None


def test_save_load_comparison_roundtrip():
    comp = {"strategy_a": {"sharpe": 1.2, "n_trades": 40}, "strategy_b": {"sharpe": 0.5, "n_trades": 10}}
    bs.save_comparison(comp)
    assert bs.load_comparison() == comp


def test_load_comparison_empty_dict_when_never_saved():
    assert bs.load_comparison() == {}


def test_save_comparison_survives_being_cleared_and_reloaded():
    bs.save_comparison({"a": {"sharpe": 1.0}})
    assert bs.load_comparison() == {"a": {"sharpe": 1.0}}
    bs.save_comparison({})
    assert bs.load_comparison() == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
