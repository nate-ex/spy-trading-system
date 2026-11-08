"""Maintains live open positions and drives entries/exits by calling the
research repo's `backtest_engine.py` decision functions DIRECTLY --
`_passes_filters`, `_entry_direction`, `_size_position`, `_open_trade`,
`_check_exit`, `_close_trade` -- unmodified, unreimplemented. This is the
whole point of the live-row-assembly approach (`signal_row.py`): the exact
same functions that decided a backtested trade decide a live one, so
there's no drift between what got backtested and what actually trades.

One symbol only (SPY), matching the rest of the research project.
"""

from __future__ import annotations

from datetime import time as _time

import backtest_engine as be
from strategy_components import StrategyConfig

from live_trading import audit_log
from live_trading.broker import AlpacaBroker

SYMBOL = "SPY"


def _parse_time(s: str) -> _time:
    h, m = s.split(":")
    return _time(int(h), int(m))


class PositionManager:
    def __init__(self, broker: AlpacaBroker, config: StrategyConfig):
        self.broker = broker
        self.config = config
        self.open_positions: list[be.OpenPosition] = []
        self._t_start = _parse_time(config.time_start)
        self._t_end = _parse_time(config.time_end)
        self._eod_t = _parse_time(config.eod_exit_time)

    def _live_account_equity(self) -> float:
        return self.broker.account()["equity"]

    def _shares_for_fraction(self, fraction: float, price: float) -> int:
        equity = self._live_account_equity()
        return max(0, int((fraction * equity) / price))

    def on_tick(self, row) -> None:
        """`row`: a `pd.Series` from `signal_row.assemble_live_signal_row`
        -- shaped exactly like a `backtest_engine.py` row, so every
        function below is called precisely as `run_backtest`'s inner loop
        already calls them, just driven by a live tick instead of a
        historical DataFrame row."""
        still_open = []
        for pos in self.open_positions:
            exit_info = be._check_exit(pos, row, self._eod_t)
            if exit_info is not None:
                self._execute_exit(pos, exit_info)
            else:
                still_open.append(pos)
        self.open_positions = still_open

        if be._passes_filters(row, self.config, self._t_start, self._t_end) and row.time.time() < self._eod_t:
            current_exposure = sum(p.fraction for p in self.open_positions)
            frac = be._size_position(row, self.config)
            if frac > 0 and current_exposure + frac <= self.config.max_total_exposure:
                direction = be._entry_direction(row, self.config)
                if direction != 0:
                    self._execute_entry(row, direction, frac)

    def force_flatten(self, row) -> None:
        """Called at/after `eod_exit_time` (or on a kill-switch command) --
        closes everything regardless of `_check_exit`'s own EOD check,
        as a second, explicit guarantee nothing is left open."""
        for pos in list(self.open_positions):
            exit_info = {"exit_time": row.time, "exit_price": float(row.close), "reason": "force_flatten"}
            self._execute_exit(pos, exit_info)
        self.open_positions = []
        self.broker.close_all()

    def _execute_entry(self, row, direction: int, fraction: float) -> None:
        shares = self._shares_for_fraction(fraction, float(row.close))
        if shares <= 0:
            return
        side = "buy" if direction == 1 else "sell"
        self.broker.submit_market_order(SYMBOL, shares, side)
        pos = be._open_trade(row, direction, fraction, self.config)
        pos.shares = shares  # noqa: attribute injected for exit sizing symmetry, not part of OpenPosition's own dataclass fields
        self.open_positions.append(pos)
        audit_log.log_event("position_opened", symbol=SYMBOL, direction=direction,
                            fraction=fraction, shares=shares, entry_price=row.close,
                            entry_reason=pos.entry_reason)

    def _execute_exit(self, pos: be.OpenPosition, exit_info: dict) -> None:
        shares = getattr(pos, "shares", None)
        side = "sell" if pos.direction == 1 else "buy"
        if shares:
            self.broker.submit_market_order(SYMBOL, shares, side)
        trade = be._close_trade(pos, exit_info, self.config)
        audit_log.log_event("position_closed", symbol=SYMBOL, **{
            k: v for k, v in trade.items() if k not in ("entry_time", "exit_time")
        }, entry_time=str(trade["entry_time"]), exit_time=str(trade["exit_time"]))
