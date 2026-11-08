"""Assembles one live "signal row" shaped exactly like
`backtest_signals.build_backtest_signals()`'s rows (research repo,
`src/backtest_signals.py`) -- so `backtest_engine.py`'s actual decision
functions (`_passes_filters`, `_entry_direction`, `_size_position`,
`_open_trade`, `_check_exit`, `_close_trade`) can be called on it directly,
unmodified. This is the one piece of genuinely new logic the live trading
loop needs: everything else (probability engines, decision rules) is
reused as-is from the research repo.

Columns produced (matches `RAW_COLS` + derived columns in
`backtest_signals.py`, restricted to what `backtest_engine.py`'s functions
actually read): `time`, `close`, `high`, `low`, `aggregate_dollar_imbalance`,
`imbalance_zscore_intraday`, `imbalance_zscore_tod`, `imbalance_reverting`,
`imbalance_accelerating`, and `p_touch_{h}`/`p_tail_{h}`/`p_upside_{h}`/
`p_downside_{h}` for h in 5/10/15/20/30min (NaN for horizons whose exit
would be after 16:00 -- both frozen engines already omit those from their
own `.predict()` output, so this just fills NaN for whatever's missing
rather than re-deriving the censoring rule).

`high`/`low` are intentionally set to the latest bar's own `close` (not
true intrabar extremes) -- the live loop polls more often than once per
minute, so there's no single well-defined "this poll's bar" the way a
historical 1-minute OHLC bar has one. Using last-known trade price for
both is a conservative simplification: a stop/target check
(`backtest_engine._check_exit`) can only trigger off a price that was
actually observed, never an interpolated/assumed wick. Documented here,
not hidden.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

SIGNAL_HORIZONS = ["5min", "10min", "15min", "20min", "30min"]


def assemble_live_signal_row(
    now: datetime,
    close: float,
    minutes_to_close: float,
    density_pred: dict,
    touch_pred: dict,
    agg_imbalance: float,
    imb_z: float,
    imb_velocity: float,
    imb_acceleration: float,
    horizons: list = SIGNAL_HORIZONS,
    imb_z_tod: float = float("nan"),
) -> pd.Series:
    """`density_pred`/`touch_pred`: per-horizon dicts from
    `Stage1DensityEngine.predict()`/`TouchProbabilityEngine.predict()`
    (via `live_stage1_feed.fetch_and_predict()` /
    `live_signal_state.touch_predictions()`) -- horizons missing from
    either (past-16:00 censoring) become NaN columns here, matching
    `backtest_signals.py`'s own censoring exactly since it's the same
    engines doing the censoring, not a re-derivation.

    `imb_velocity`/`imb_acceleration`: from
    `live_signal_state.live_imbalance_velocity_acceleration()`.
    `imb_z_tod`: from `live_signal_state.live_imbalance_zscore_tod()` --
    needed here (not just for display) so a deployed strategy with
    `zscore_source="time_of_day_normalized"` (see
    `strategy_components.py`) actually has this column to read; defaults
    to NaN so a caller not yet passing it degrades to "not warmed up"
    instead of an AttributeError, matching every other optional-signal
    convention in this module.
    """
    data: dict = {
        "time": now,
        "close": float(close),
        "high": float(close),
        "low": float(close),
        "minutes_to_close": float(minutes_to_close),
        "aggregate_dollar_imbalance": float(agg_imbalance),
        "imbalance_zscore_intraday": float(imb_z) if pd.notna(imb_z) else float("nan"),
        "imbalance_zscore_tod": float(imb_z_tod) if pd.notna(imb_z_tod) else float("nan"),
        "imbalance_reverting": bool((agg_imbalance * imb_velocity) < 0) if pd.notna(imb_velocity) else False,
        "imbalance_accelerating": bool((agg_imbalance * imb_acceleration) > 0) if pd.notna(imb_acceleration) else False,
    }
    for h in horizons:
        d = density_pred.get(h, {})
        t = touch_pred.get(h, {})
        data[f"p_tail_{h}"] = float(d["P_tail"]) if "P_tail" in d else float("nan")
        data[f"p_upside_{h}"] = float(d["P_upside"]) if "P_upside" in d else float("nan")
        data[f"p_downside_{h}"] = float(d["P_downside"]) if "P_downside" in d else float("nan")
        data[f"p_touch_{h}"] = float(t["P_touch"]) if "P_touch" in t else float("nan")
    return pd.Series(data)
