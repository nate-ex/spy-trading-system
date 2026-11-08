"""Core strategy simulation engine — bar-by-bar (1-minute), day-by-day,
supports overlapping/pyramided positions. Operates only on
`backtest_signals.py`'s precomputed development-only signal table.

No-lookahead conventions (mirrors `q000_backtest.py`'s conventions where
applicable):
  - A signal row's own `close` is the entry price -- the price known at
    the moment its probability/imbalance state became available.
  - A position is NEVER exit-checked on its own entry bar. Profit-target/
    stop-loss touches are evaluated against EACH SUBSEQUENT bar's
    high/low, starting the bar after entry -- using the entry bar's own
    high/low would leak information from before the entry decision was
    even made (the bar's high could have printed before its close).
  - If both a profit target and a stop loss could have been touched
    within the same bar, the stop loss is assumed to have triggered
    first (the conservative, standard backtest convention -- intrabar
    ordering is not observable from 1-minute OHLC).
  - Every position is force-closed at `eod_exit_time` (default 16:00,
    matching the project's hard regular-hours-only constraint) even if no
    other exit rule fired.
  - Position returns are additive (fixed-fraction-of-bankroll, summed not
    compounded) — the same simplification `q000_backtest.py` uses.

Direction: `direction_mode="both"` takes its direction from
`cfg.direction_source` at the entry bar -- either the imbalance level's
own sign ("raw_imbalance") or the chosen z-score's sign ("zscore_sign",
`cfg.zscore_source`) -- this project's own findings (Q007-Q009) are that
neither probability engine implies a direction, so these two imbalance-
derived signals are the only directional features available. They can
disagree: a large-but-typical-for-today imbalance reads raw_imbalance>0
with zscore~0, so which one drives direction is a real strategy choice,
not an implementation detail.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time as _time

import numpy as np
import pandas as pd

from strategy_components import StrategyConfig


def _parse_time(s: str) -> _time:
    h, m = s.split(":")
    return _time(int(h), int(m))


@dataclass
class OpenPosition:
    entry_time: pd.Timestamp
    entry_price: float
    direction: int
    fraction: float
    entry_reason: str
    target_price: float | None
    stop_price: float | None
    max_exit_time: pd.Timestamp | None


@dataclass
class BacktestResult:
    config: StrategyConfig
    trades: pd.DataFrame   # one row per closed trade
    equity: pd.DataFrame   # one row per minute: time, equity, exposure


_ZSCORE_SOURCE_COLS = {
    "intraday_expanding": "imbalance_zscore_intraday",
    "time_of_day_normalized": "imbalance_zscore_tod",
    "imbalance_price_spread": "imbalance_price_spread",
}


def _zscore_col(cfg: StrategyConfig) -> str:
    """Which column `min_abs_zscore`/`direction_source="zscore_sign"` read,
    per `cfg.zscore_source` -- "intraday_expanding" is
    `imbalance_zscore_intraday` (same-day-only std, what existing
    strategies were validated against); "time_of_day_normalized" is
    `imbalance_zscore_tod` (cross-day, time-of-day-matched std);
    "imbalance_price_spread" is imbalance z-scored net of SPY's own
    already-realized move (see `relative_state_features.py`)."""
    return _ZSCORE_SOURCE_COLS[cfg.zscore_source]


def _passes_filters(row, cfg: StrategyConfig, t_start: _time, t_end: _time) -> bool:
    tod = row.time.time()
    if not (t_start <= tod <= t_end):
        return False
    prob_col = f"{cfg.prob_metric}_{cfg.entry_horizon}"
    p = getattr(row, prob_col)
    if pd.isna(p) or p < cfg.prob_threshold:
        return False
    if pd.isna(row.aggregate_dollar_imbalance):
        return False
    if cfg.min_abs_zscore > 0:
        z = getattr(row, _zscore_col(cfg))
        if pd.isna(z) or abs(z) < cfg.min_abs_zscore:
            return False
    if cfg.imbalance_direction == "positive" and not (row.aggregate_dollar_imbalance > 0):
        return False
    if cfg.imbalance_direction == "negative" and not (row.aggregate_dollar_imbalance < 0):
        return False
    if cfg.imbalance_state == "reverting" and not row.imbalance_reverting:
        return False
    if cfg.imbalance_state == "accelerating" and not row.imbalance_accelerating:
        return False
    return True


def _entry_direction(row, cfg: StrategyConfig) -> int:
    if cfg.direction_mode == "long_only":
        d = 1
    elif cfg.direction_mode == "short_only":
        d = -1
    elif cfg.direction_source == "zscore_sign":
        z = getattr(row, _zscore_col(cfg))
        d = 0 if pd.isna(z) else int(np.sign(z))
    else:
        d = int(np.sign(row.aggregate_dollar_imbalance))
    if d == 0:
        return 0
    if cfg.require_imbalance_agreement and cfg.direction_mode != "both":
        imb_sign = int(np.sign(row.aggregate_dollar_imbalance))
        if imb_sign != 0 and imb_sign != d:
            return 0
    return d


def _size_position(row, cfg: StrategyConfig) -> float:
    if cfg.sizing_mode == "fixed":
        frac = cfg.fixed_fraction
    else:
        prob_col = f"{cfg.prob_metric}_{cfg.entry_horizon}"
        p = getattr(row, prob_col)
        confidence = max(float(p) - 0.5, 0.0) * 2.0  # 0 at p=0.5, 1 at p=1.0
        frac = cfg.confidence_scale * confidence
    return float(max(0.0, min(frac, cfg.max_position_fraction)))


def _open_trade(row, direction: int, fraction: float, cfg: StrategyConfig) -> OpenPosition:
    entry_price = float(row.close)
    target = None
    if cfg.profit_target_bps is not None:
        target = entry_price * (1 + direction * cfg.profit_target_bps / 1e4)
    stop = None
    if cfg.stop_loss_bps is not None:
        stop = entry_price * (1 - direction * cfg.stop_loss_bps / 1e4)
    max_exit = None
    if cfg.holding_period_min is not None:
        max_exit = row.time + pd.Timedelta(minutes=cfg.holding_period_min)
    reason_bits = [f"{cfg.prob_metric}_{cfg.entry_horizon}>={cfg.prob_threshold}"]
    if cfg.imbalance_direction != "any":
        reason_bits.append(f"imbalance {cfg.imbalance_direction}")
    if cfg.imbalance_state != "any":
        reason_bits.append(f"imbalance {cfg.imbalance_state}")
    if cfg.min_abs_zscore > 0:
        reason_bits.append(f"|z|>={cfg.min_abs_zscore}")
    return OpenPosition(
        entry_time=row.time, entry_price=entry_price, direction=direction, fraction=fraction,
        entry_reason="; ".join(reason_bits), target_price=target, stop_price=stop, max_exit_time=max_exit,
    )


def _check_exit(pos: OpenPosition, row, eod_t: _time) -> dict | None:
    """Returns {'exit_time','exit_price','reason'} if `pos` should close on
    this bar, else None. Checked in a fixed priority: EOD force-close,
    then stop loss (conservative — assumed first if both trigger),
    then profit target, then time-based holding-period exit."""
    tod = row.time.time()
    if tod >= eod_t:
        return {"exit_time": row.time, "exit_price": float(row.close), "reason": "eod"}
    if pos.stop_price is not None:
        hit = (row.low <= pos.stop_price) if pos.direction == 1 else (row.high >= pos.stop_price)
        if hit:
            return {"exit_time": row.time, "exit_price": float(pos.stop_price), "reason": "stop_loss"}
    if pos.target_price is not None:
        hit = (row.high >= pos.target_price) if pos.direction == 1 else (row.low <= pos.target_price)
        if hit:
            return {"exit_time": row.time, "exit_price": float(pos.target_price), "reason": "profit_target"}
    if pos.max_exit_time is not None and row.time >= pos.max_exit_time:
        return {"exit_time": row.time, "exit_price": float(row.close), "reason": "holding_period"}
    return None


def _close_trade(pos: OpenPosition, exit_info: dict, cfg: StrategyConfig) -> dict:
    raw_ret = pos.direction * (exit_info["exit_price"] / pos.entry_price - 1)
    cost = cfg.cost_bps / 1e4  # round-trip, applied once per closed trade
    pnl = pos.fraction * (raw_ret - cost)
    return {
        "entry_time": pos.entry_time, "exit_time": exit_info["exit_time"],
        "direction": pos.direction, "fraction": pos.fraction,
        "entry_price": pos.entry_price, "exit_price": exit_info["exit_price"],
        "raw_return": raw_ret, "pnl": pnl, "exit_reason": exit_info["reason"],
        "entry_reason": pos.entry_reason,
        "holding_minutes": (exit_info["exit_time"] - pos.entry_time).total_seconds() / 60.0,
    }


def run_backtest(signals: pd.DataFrame, config: StrategyConfig) -> BacktestResult:
    t_start = _parse_time(config.time_start)
    t_end = _parse_time(config.time_end)
    eod_t = _parse_time(config.eod_exit_time)

    trades: list[dict] = []
    equity_rows: list[dict] = []
    realized = 0.0

    for _date, day in signals.groupby("date", sort=True):
        day = day.sort_values("time")
        open_positions: list[OpenPosition] = []

        for row in day.itertuples(index=False):
            still_open = []
            for pos in open_positions:
                exit_info = _check_exit(pos, row, eod_t)
                if exit_info is not None:
                    trade = _close_trade(pos, exit_info, config)
                    realized += trade["pnl"]
                    trades.append(trade)
                else:
                    still_open.append(pos)
            open_positions = still_open

            if _passes_filters(row, config, t_start, t_end) and row.time.time() < eod_t:
                current_exposure = sum(p.fraction for p in open_positions)
                frac = _size_position(row, config)
                if frac > 0 and current_exposure + frac <= config.max_total_exposure:
                    direction = _entry_direction(row, config)
                    if direction != 0:
                        open_positions.append(_open_trade(row, direction, frac, config))

            price = float(row.close)
            unrealized = sum(p.fraction * p.direction * (price / p.entry_price - 1) for p in open_positions)
            exposure = sum(p.fraction for p in open_positions)
            equity_rows.append({"time": row.time, "equity": realized + unrealized, "exposure": exposure})

        if open_positions:
            last_row = day.iloc[-1]
            for pos in open_positions:
                exit_info = {"exit_time": last_row["time"], "exit_price": float(last_row["close"]), "reason": "eod"}
                trade = _close_trade(pos, exit_info, config)
                realized += trade["pnl"]
                trades.append(trade)
            # This is a safety net for data that ends before `eod_exit_time`
            # (e.g. a half day) -- the per-bar loop above already wrote an
            # equity row at `last_row`'s timestamp (unrealized); update it
            # in place now that it's realized, rather than appending a
            # second row at the same timestamp (which would make the
            # equity curve non-unique-indexed).
            if equity_rows and equity_rows[-1]["time"] == last_row["time"]:
                equity_rows[-1]["equity"] = realized
                equity_rows[-1]["exposure"] = 0.0
            else:
                equity_rows.append({"time": last_row["time"], "equity": realized, "exposure": 0.0})

    trades_df = pd.DataFrame(trades, columns=[
        "entry_time", "exit_time", "direction", "fraction", "entry_price", "exit_price",
        "raw_return", "pnl", "exit_reason", "entry_reason", "holding_minutes",
    ])
    equity_df = pd.DataFrame(equity_rows, columns=["time", "equity", "exposure"])
    return BacktestResult(config=config, trades=trades_df, equity=equity_df)
