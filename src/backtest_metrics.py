"""Performance metrics for a `BacktestResult` (backtest_engine.py).

P&L is expressed in fixed-fraction-of-bankroll return units (additive,
not compounded -- see backtest_engine.py's docstring), the same
convention `q000_backtest.py` uses -- so `total_pnl` is a return, not a
dollar figure, unless the caller multiplies by an actual bankroll size.

Sharpe/Sortino are computed on DAILY aggregated P&L (one number per
trading day), annualized by sqrt(252), matching `q000_backtest.
sharpe_ratio`'s convention -- per-trade Sharpe would be a different,
non-comparable number (trades aren't independent, identically-timed
draws the way daily P&L approximately is).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _daily_pnl(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    d = trades.copy()
    d["day"] = pd.to_datetime(d["exit_time"]).dt.date
    return d.groupby("day")["pnl"].sum()


def sharpe_ratio(daily_pnl: pd.Series) -> float:
    if len(daily_pnl) < 2 or daily_pnl.std(ddof=1) == 0:
        return 0.0
    return float(daily_pnl.mean() / daily_pnl.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(daily_pnl: pd.Series) -> float:
    downside = daily_pnl[daily_pnl < 0]
    if len(daily_pnl) < 2 or len(downside) == 0 or downside.std(ddof=1) == 0:
        return 0.0
    return float(daily_pnl.mean() / downside.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity: pd.Series) -> float:
    """Additive (return-unit) drawdown: min(equity - running_max(equity))."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = equity - running_max
    return float(dd.min())


def profit_factor(trades: pd.DataFrame) -> float:
    if trades.empty:
        return float("nan")
    wins = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    losses = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
    if losses == 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / losses)


def pnl_by_time_of_day(trades: pd.DataFrame, bucket_minutes: int = 15) -> pd.DataFrame:
    """Sum/mean/count of P&L bucketed by the entry time's clock time (not
    date) -- e.g. all trades entered in the 14:00-14:15 bucket, pooled
    across every day in the run."""
    if trades.empty:
        return pd.DataFrame(columns=["bucket", "total_pnl", "avg_pnl", "n_trades"])
    d = trades.copy()
    et = pd.to_datetime(d["entry_time"])
    minutes_since_midnight = et.dt.hour * 60 + et.dt.minute
    bucket_start = (minutes_since_midnight // bucket_minutes) * bucket_minutes
    d["bucket"] = bucket_start.map(lambda m: f"{m // 60:02d}:{m % 60:02d}")
    g = d.groupby("bucket")["pnl"].agg(total_pnl="sum", avg_pnl="mean", n_trades="count")
    return g.reset_index().sort_values("bucket")


def buy_hold_benchmark(signals: pd.DataFrame, d1, d2, time_start: str = "14:00",
                       time_end: str = "16:00") -> dict:
    """Passive long-only SPY benchmark: buy at `time_start`, hold to
    `time_end`, every trading day in [d1, d2] -- additive daily returns
    (not compounded), the same convention `compute_metrics()` uses for a
    strategy's own P&L, so this Sharpe/return is directly comparable,
    dropped straight into the same METRIC_COLS the Compare tab already
    sorts on (`dashboard_trading.py`).

    This is the "SPY 2-4pm buy&hold" reference this project's own
    research already uses (Q008, STATISTICAL_PROCESS.md) to tell whether
    a strategy's return is actual alpha or just beta drift -- added here
    as a first-class, always-available comparison rather than a one-off
    number quoted in a writeup.
    """
    from datetime import time as _time

    def _parse(s: str) -> _time:
        h, m = s.split(":")
        return _time(int(h), int(m))

    t_start, t_end = _parse(time_start), _parse(time_end)
    dates = signals["date"].dt.date
    sub = signals[(dates >= d1) & (dates <= d2)]

    daily_rows = []
    for date, day in sub.groupby(sub["date"].dt.date):
        tod = day["time"].dt.time
        win = day[(tod >= t_start) & (tod <= t_end)].sort_values("time")
        if len(win) < 2:
            continue
        entry_price = float(win["close"].iloc[0])
        exit_price = float(win["close"].iloc[-1])
        daily_rows.append({"date": date, "pnl": exit_price / entry_price - 1})

    empty = {"n_trades": 0, "total_pnl": 0.0, "avg_pnl_per_trade": float("nan"),
            "win_rate": float("nan"), "max_drawdown": 0.0, "sharpe": 0.0, "sortino": 0.0,
            "profit_factor": float("nan"), "avg_holding_minutes": float("nan"), "avg_exposure": 1.0}
    if not daily_rows:
        return empty

    daily = pd.Series([r["pnl"] for r in daily_rows], index=[r["date"] for r in daily_rows])
    cum = daily.cumsum()
    dd = float((cum - cum.cummax()).min())
    wins = daily[daily > 0].sum(); losses = -daily[daily < 0].sum()
    pf = float(wins / losses) if losses > 0 else (float("inf") if wins > 0 else float("nan"))
    holding_minutes = (_time_to_minutes(t_end) - _time_to_minutes(t_start))

    return {
        "n_trades": int(len(daily)),  # "trades" here = trading days held, for column compatibility
        "total_pnl": float(daily.sum()),
        "avg_pnl_per_trade": float(daily.mean()),
        "win_rate": float((daily > 0).mean()),
        "max_drawdown": dd,
        "sharpe": sharpe_ratio(daily),
        "sortino": sortino_ratio(daily),
        "profit_factor": pf,
        "avg_holding_minutes": float(holding_minutes),
        "avg_exposure": 1.0,  # fully invested every session, by construction
    }


def _time_to_minutes(t) -> int:
    return t.hour * 60 + t.minute


def compute_metrics(result) -> dict:
    """`result`: a backtest_engine.BacktestResult. Returns the full metrics
    dict the dashboard displays per strategy."""
    trades = result.trades
    equity = result.equity

    if trades.empty:
        return {
            "n_trades": 0, "total_pnl": 0.0, "avg_pnl_per_trade": float("nan"),
            "win_rate": float("nan"), "max_drawdown": 0.0, "sharpe": 0.0, "sortino": 0.0,
            "profit_factor": float("nan"), "avg_holding_minutes": float("nan"),
            "avg_exposure": float(equity["exposure"].mean()) if not equity.empty else 0.0,
        }

    daily = _daily_pnl(trades)
    wins = trades[trades["pnl"] > 0]

    return {
        "n_trades": int(len(trades)),
        "total_pnl": float(trades["pnl"].sum()),
        "avg_pnl_per_trade": float(trades["pnl"].mean()),
        "win_rate": float((trades["pnl"] > 0).mean()),
        "max_drawdown": max_drawdown(equity["equity"]) if not equity.empty else 0.0,
        "sharpe": sharpe_ratio(daily),
        "sortino": sortino_ratio(daily),
        "profit_factor": profit_factor(trades),
        "avg_holding_minutes": float(trades["holding_minutes"].mean()),
        "avg_exposure": float(equity["exposure"].mean()) if not equity.empty else 0.0,
        "n_wins": int(len(wins)), "n_losses": int((trades["pnl"] < 0).sum()),
        "n_days_active": int(daily.shape[0]),
    }
