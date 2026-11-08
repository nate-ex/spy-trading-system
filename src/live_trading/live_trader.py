"""The persistent background live-trading loop. Runs as its own process
(`run_live_trading.py`), independent of the Streamlit dashboard -- a
closed browser tab must never stop open positions from being managed.

Every poll (synchronous, not asyncio -- there's nothing to run
concurrently here, just one sequential sequence of REST calls each cycle,
matching this project's other simple polling feeds):
  1. Check `command.json` FIRST, before anything else. A pending "kill"
     always wins, even mid-strategy, even before the rest of this poll's
     work happens.
  2. If a strategy is deployed and running (and it's within regular
     trading hours, 14:00-16:00 ET -- this project's hard regular-hours-
     only mandate, never relaxed here): fetch live SPY bars + Stage-1/
     touch predictions + TradeMas imbalance, assemble a signal row, hand
     it to the `PositionManager`.
  3. Force-flatten everything at/after the strategy's `eod_exit_time` (or
     16:00, whichever is earlier) -- a second, explicit guarantee on top
     of `backtest_engine._check_exit`'s own EOD check.
  4. Write `status.json` regardless of state, so the dashboard always has
     a fresh liveness signal.
"""

from __future__ import annotations

import time as time_mod
from datetime import datetime, time as dtime

import pytz

from live_trading import ipc
from live_trading.broker import AlpacaBroker
from live_trading.position_manager import PositionManager, SYMBOL
from live_trading.signal_row import assemble_live_signal_row, SIGNAL_HORIZONS

EST = pytz.timezone("US/Eastern")
TRADING_WINDOW_START = dtime(14, 0)
HARD_STOP = dtime(16, 0)  # project-wide regular-hours-only mandate -- never relaxed
POLL_SECONDS = 1


def _now_et() -> datetime:
    return datetime.now(EST).replace(tzinfo=None)


class LiveTrader:
    def __init__(self, poll_seconds: int = POLL_SECONDS):
        self.poll_seconds = poll_seconds
        self.state = "idle"
        self.active_config_path: "str | None" = None
        self.mode: "str | None" = None
        self.manager: "PositionManager | None" = None
        self.paused = False
        self._last_command_id: "str | None" = None

    # ---- command handling ----
    def _handle_command(self) -> bool:
        """Returns True if a "kill" was processed this poll (caller should
        stop the whole process after this)."""
        command = ipc.read_pending_command(self._last_command_id)
        if command is None:
            return False
        self._last_command_id = command["id"]
        action = command["action"]

        if action == "kill":
            if self.manager is not None:
                row = self._current_row_for_flatten()
                if row is not None:
                    self.manager.force_flatten(row)
                else:
                    self.manager.broker.close_all()
            self.state = "halted"
            self._write_status()
            return True

        if action == "deploy":
            import backtest_storage

            config = backtest_storage.load_config(command["config_name"])
            self.mode = command.get("mode", "paper")
            self._deploy(config, self.mode)
            self.active_config_path = command["config_name"]
        elif action == "pause":
            self.paused = True
        elif action == "resume":
            self.paused = False

        self._write_status()
        return False

    def _deploy(self, config, mode: str) -> None:
        import os

        api_key = os.environ["ALPACA_API_KEY"]
        secret_key = os.environ["ALPACA_SECRET_KEY"]
        broker = AlpacaBroker(api_key, secret_key, paper=(mode == "paper"))
        self.manager = PositionManager(broker, config)
        self.state = "running"
        self.paused = False

    def _current_row_for_flatten(self):
        """Best-effort last-known row for a force-flatten's exit-price
        bookkeeping -- if fetching a fresh price fails (e.g. mid-kill,
        network hiccup), `force_flatten` still needs SOME row shape to
        close positions against; falls back to closing via the broker
        directly (see `_handle_command`'s "kill" branch) if this raises."""
        try:
            return self._poll_live_state()
        except Exception:
            return None

    # ---- live state ----
    def _poll_live_state(self):
        import live_stage1_feed as lsf
        from stage1_density_engine import Stage1DensityEngine
        from touch_probability_engine import TouchProbabilityEngine
        from live_feeds import TradeMasImbalanceFeed, aggregate_dollar_imbalance
        from live_signal_state import (
            touch_predictions, load_live_imbalance_history,
            live_imbalance_zscore_intraday, live_imbalance_zscore_tod,
            live_imbalance_velocity_acceleration,
            append_live_imbalance_snapshot,
        )

        density_engine = Stage1DensityEngine.load()
        touch_engine = TouchProbabilityEngine.load()

        # source="alpaca" -- this loop must never attempt Bloomberg, on
        # this machine or any other it might run on. A Bloomberg Terminal
        # session is specific to the research-desk machine, not something
        # a personal trading process should ever try or depend on.
        result = lsf.fetch_and_predict(density_engine, ticker=SYMBOL, tau_bps=10.0, source="alpaca")
        now = result["features"]["time"]
        now = now if isinstance(now, datetime) else datetime.combine(datetime.today(), now)
        close = result["bars"]["close"].iloc[-1] if not result["bars"].empty else float("nan")

        try:
            ticks = TradeMasImbalanceFeed().fetch_once()
        except Exception:
            ticks = []
        agg_imbalance = aggregate_dollar_imbalance(ticks) if ticks else 0.0

        prior_hist = load_live_imbalance_history(now.date())
        imb_z = live_imbalance_zscore_intraday(agg_imbalance, prior_hist)
        imb_z_tod = live_imbalance_zscore_tod(agg_imbalance, prior_hist, now.time())
        imb_velocity, imb_accel = live_imbalance_velocity_acceleration(agg_imbalance, prior_hist)
        # Only record a snapshot when TradeMas actually returned ticks --
        # outside its active window `ticks` is empty and `agg_imbalance` is
        # a placeholder 0.0, not a genuine reading. Appending it anyway
        # would flood the day's history with fake zeros that drag the
        # z-score's reference mean down and its reference std toward 0,
        # making later real readings register as absurd outlier z-scores
        # regardless of their actual size (see dashboard_view.py's
        # matching fix for the full mechanism).
        if ticks:
            append_live_imbalance_snapshot(now, agg_imbalance)

        touch_pred = touch_predictions(touch_engine, result["features"], now.time(), 10.0) \
            if result["warmed_up"] else {}
        density_pred = result["prediction"] if result["warmed_up"] else {}

        return assemble_live_signal_row(
            now=now, close=close, minutes_to_close=result["features"].get("minutes_to_close", float("nan")),
            density_pred=density_pred, touch_pred=touch_pred,
            agg_imbalance=agg_imbalance, imb_z=imb_z, imb_z_tod=imb_z_tod,
            imb_velocity=imb_velocity, imb_acceleration=imb_accel,
            horizons=SIGNAL_HORIZONS,
        )

    # ---- status ----
    def _write_status(self) -> None:
        positions, equity, realized_pnl, unrealized_pnl, last_error = [], 0.0, 0.0, 0.0, None
        if self.manager is not None:
            try:
                acct = self.manager.broker.account()
                equity = acct["equity"]
                positions = self.manager.broker.positions()
                unrealized_pnl = sum(p["unrealized_pl"] for p in positions)
            except Exception as exc:
                last_error = str(exc)
        ipc.write_status(
            state=self.state, active_config=self.active_config_path, mode=self.mode,
            positions=positions, equity=equity, realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl, last_error=last_error,
        )

    # ---- main loop ----
    def run_forever(self) -> None:
        while True:
            killed = self._handle_command()
            if killed:
                print("[live_trader] kill command processed -- stopping.")
                return

            if self.state == "running" and self.manager is not None:
                now_t = _now_et().time()
                try:
                    if now_t >= self.manager._eod_t or now_t >= HARD_STOP:
                        row = self._poll_live_state()
                        self.manager.force_flatten(row)
                        self.state = "idle"
                    elif TRADING_WINDOW_START <= now_t < HARD_STOP:
                        row = self._poll_live_state()
                        self.manager.on_tick(row) if not self.paused else \
                            self._exits_only(row)
                except Exception as exc:
                    print(f"[live_trader] poll failed: {exc}")

            self._write_status()
            time_mod.sleep(self.poll_seconds)

    def _exits_only(self, row) -> None:
        """Paused: manage existing positions' exits (stop/target/EOD),
        never open new ones -- a position already open must still be
        looked after while paused, not silently abandoned."""
        still_open = []
        import backtest_engine as be
        for pos in self.manager.open_positions:
            exit_info = be._check_exit(pos, row, self.manager._eod_t)
            if exit_info is not None:
                self.manager._execute_exit(pos, exit_info)
            else:
                still_open.append(pos)
        self.manager.open_positions = still_open
