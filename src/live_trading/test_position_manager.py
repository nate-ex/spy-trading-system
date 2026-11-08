"""Unit tests for PositionManager -- verifies it drives real order calls
on entry/exit decisions made by the research repo's actual
`backtest_engine.py` functions (not a reimplementation), using a fake
broker (no live network)."""

from datetime import datetime

import pytest

from live_trading import audit_log
from live_trading.position_manager import PositionManager
from live_trading.signal_row import assemble_live_signal_row
from strategy_components import StrategyConfig


@pytest.fixture(autouse=True)
def _isolated_log_dir(tmp_path, monkeypatch):
    """PositionManager logs position open/close events unconditionally
    (independent of which broker is used) -- without this, tests would
    silently write fake position events into the repo's real, COMMITTED
    audit log."""
    monkeypatch.setattr(audit_log, "LOG_DIR", tmp_path)
    yield


class _FakeBroker:
    def __init__(self, equity=100_000.0):
        self._equity = equity
        self.orders = []
        self.closed_all_called = False

    def account(self):
        return {"equity": self._equity}

    def submit_market_order(self, symbol, qty, side, time_in_force="day"):
        self.orders.append((symbol, qty, side))
        return {"symbol": symbol, "qty": qty, "side": side, "status": "filled"}

    def close_all(self):
        self.closed_all_called = True
        return []


def _row(now, close, agg_imbalance=1e9, p_touch_5min=0.9):
    return assemble_live_signal_row(
        now=now, close=close, minutes_to_close=(datetime(2026, 7, 21, 16, 0) - now).total_seconds() / 60.0,
        density_pred={}, touch_pred={"5min": {"P_touch": p_touch_5min}},
        agg_imbalance=agg_imbalance, imb_z=2.0, imb_velocity=1e7, imb_acceleration=1e6,
    )


def _config(**overrides) -> StrategyConfig:
    base = dict(prob_metric="p_touch", entry_horizon="5min", prob_threshold=0.5,
               direction_mode="both", fixed_fraction=0.5, sizing_mode="fixed")
    base.update(overrides)
    return StrategyConfig(**base)


def test_on_tick_opens_a_position_and_places_an_order():
    broker = _FakeBroker(equity=100_000.0)
    manager = PositionManager(broker, _config())
    row = _row(datetime(2026, 7, 21, 14, 30), close=500.0)

    manager.on_tick(row)

    assert len(manager.open_positions) == 1
    assert len(broker.orders) == 1
    symbol, qty, side = broker.orders[0]
    assert symbol == "SPY" and side == "buy"
    # fraction=0.5, equity=100k, price=500 -> 100 shares
    assert qty == 100


def test_on_tick_does_not_enter_when_probability_below_threshold():
    broker = _FakeBroker()
    manager = PositionManager(broker, _config(prob_threshold=0.95))
    row = _row(datetime(2026, 7, 21, 14, 30), close=500.0, p_touch_5min=0.6)

    manager.on_tick(row)

    assert manager.open_positions == []
    assert broker.orders == []


def test_on_tick_closes_on_stop_loss_and_places_exit_order():
    broker = _FakeBroker(equity=100_000.0)
    config = _config(stop_loss_bps=50.0)  # 0.5%
    manager = PositionManager(broker, config)
    entry_row = _row(datetime(2026, 7, 21, 14, 30), close=500.0)
    manager.on_tick(entry_row)
    assert len(manager.open_positions) == 1
    assert len(broker.orders) == 1

    # Price drops below the stop (500 * (1 - 0.005) = 497.5) on a later bar.
    # p_touch below threshold so this same tick doesn't ALSO immediately
    # re-enter right after the exit (on_tick checks exits then entries on
    # every call, matching backtest_engine.run_backtest's own per-row
    # ordering exactly -- not a bug, just needs isolating in this test).
    exit_row = _row(datetime(2026, 7, 21, 14, 35), close=497.0, p_touch_5min=0.1)
    manager.on_tick(exit_row)

    assert manager.open_positions == []
    assert len(broker.orders) == 2
    symbol, qty, side = broker.orders[1]
    assert side == "sell" and qty == 100


def test_force_flatten_closes_everything_and_calls_broker_close_all():
    broker = _FakeBroker(equity=100_000.0)
    manager = PositionManager(broker, _config())
    entry_row = _row(datetime(2026, 7, 21, 14, 30), close=500.0)
    manager.on_tick(entry_row)
    assert len(manager.open_positions) == 1

    flatten_row = _row(datetime(2026, 7, 21, 16, 0), close=505.0)
    manager.force_flatten(flatten_row)

    assert manager.open_positions == []
    assert broker.closed_all_called is True


def test_on_tick_respects_max_total_exposure():
    broker = _FakeBroker(equity=100_000.0)
    config = _config(fixed_fraction=0.6, max_total_exposure=1.0)
    manager = PositionManager(broker, config)
    row1 = _row(datetime(2026, 7, 21, 14, 30), close=500.0)
    manager.on_tick(row1)
    assert len(manager.open_positions) == 1

    # A second entry at 0.6 fraction would push total exposure to 1.2 >
    # max_total_exposure=1.0 -- must be rejected.
    row2 = _row(datetime(2026, 7, 21, 14, 31), close=500.0)
    manager.on_tick(row2)
    assert len(manager.open_positions) == 1
    assert len(broker.orders) == 1
