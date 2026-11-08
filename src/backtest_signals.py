"""Precompute per-minute decision-time signals for the strategy backtest
framework (`backtest_engine.py`). DEVELOPMENT DATA ONLY -- this whole
framework is exploratory strategy research, not a pre-registered
hypothesis test, and per RESEARCH_CHARTER.md/PRD.md the held-out test set
is touched at most once, after methodology is finalized. Never call
`data_access.load_holdout_test()` from anywhere in this module or its
callers.

One row per development-set 1-minute bar in [14:00, 16:00] (the same
window both frozen engines operate in), carrying:
  - raw OHLC + time keys
  - both frozen engines' probability outputs per horizon (5/10/15/20/30min):
    `p_touch_{h}` (touch/first-passage, `touch_probability_engine.py`) and
    `p_tail_{h}`/`p_upside_{h}`/`p_downside_{h}` (terminal-return,
    `stage1_density_engine.py`) -- NaN wherever that horizon's exit would
    be after 16:00, mirroring each engine's own predict()-time censoring
    (t + h <= 16:00).
  - imbalance state: the existing `aggregate_dollar_imbalance`/
    `imbalance_zscore_intraday`/`imbalance_velocity`/`imbalance_acceleration`
    columns, plus two derived booleans built from them --
    `imbalance_reverting` (True if velocity points opposite the current
    level, i.e. |imbalance| shrinking) and `imbalance_accelerating` (True
    if acceleration agrees in sign with the current level, i.e. |imbalance|
    growing at an increasing rate).

IMPORTANT (do not treat probability as a directional win-rate): neither
`p_touch_h` nor `p_tail_h` is a directional forecast -- this project's own
findings (Q007/Q008/Q009) are that conditional mean AND conditional skew
are a martingale in regular hours. These are MAGNITUDE/dispersion
probabilities only ("how likely is a big move"), never "which way." Any
strategy that needs a direction must get it from the imbalance sign (the
only directional feature this project has), not from either engine.

Cached to `data/processed/backtest_signals_dev.parquet` since it's
expensive-ish to recompute (two engines x 3 horizons over ~90k rows) and
the sweep framework may call it many times per session.
"""

from __future__ import annotations

from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

import engine_versioning
import q007_distribution as q7
from stage1_density_engine import Stage1DensityEngine, BASE_SCALE_COLS, _add_time_features, MOC
from touch_probability_engine import TouchProbabilityEngine

WINDOW_START = time(14, 0)
SIGNAL_HORIZONS = ["5min", "10min", "15min", "20min", "30min"]
DEFAULT_TAU_BPS = 10.0
CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "backtest_signals_dev.parquet"

RAW_COLS = [
    "time", "date", "open", "high", "low", "close", "minutes_to_close",
    "aggregate_dollar_imbalance", "imbalance_zscore_intraday", "imbalance_zscore_tod",
    "imbalance_price_spread",
    "imbalance_velocity", "imbalance_acceleration", "imbalance_stale",
]


def _exit_within_moc(ts: pd.Series, h: str) -> np.ndarray:
    exit_dt = ts + pd.Timedelta(minutes=q7.HORIZON_MINUTES[h])
    return (exit_dt.dt.time <= MOC).to_numpy()


def build_backtest_signals(dev: pd.DataFrame, tau_bps: float = DEFAULT_TAU_BPS) -> pd.DataFrame:
    t = pd.to_datetime(dev["time"])
    win = dev[(t.dt.time >= WINDOW_START) & (t.dt.time <= MOC)].copy()
    win = _add_time_features(win).sort_values("time").reset_index(drop=True)
    ts = pd.to_datetime(win["time"])

    touch_engine = TouchProbabilityEngine.load()
    terminal_engine = Stage1DensityEngine.load()
    tau = tau_bps / 1e4
    has_state = win[BASE_SCALE_COLS].notna().all(axis=1).to_numpy()

    for h in SIGNAL_HORIZONS:
        mask = _exit_within_moc(ts, h) & has_state

        touch = touch_engine.touch_probs(h, win, tau)
        win[f"p_touch_{h}"] = np.where(mask, touch["P_touch"], np.nan)

        term = terminal_engine.tail_probs(h, win, tau)
        win[f"p_tail_{h}"] = np.where(mask, term["P_tail"], np.nan)
        win[f"p_upside_{h}"] = np.where(mask, term["P_upside"], np.nan)
        win[f"p_downside_{h}"] = np.where(mask, term["P_downside"], np.nan)

    imb = win["aggregate_dollar_imbalance"]
    vel = win["imbalance_velocity"]
    acc = win["imbalance_acceleration"]
    win["imbalance_reverting"] = (imb * vel) < 0
    win["imbalance_accelerating"] = (imb * acc) > 0

    cols = RAW_COLS + ["imbalance_reverting", "imbalance_accelerating"]
    for h in SIGNAL_HORIZONS:
        cols += [f"p_touch_{h}", f"p_tail_{h}", f"p_upside_{h}", f"p_downside_{h}"]
    return win[cols].reset_index(drop=True)


def load_or_build_signals(force: bool = False) -> pd.DataFrame:
    """Development-only. Loads the cache if present AND built from the
    currently-loaded engines (see `engine_versioning.is_stale` -- a stale
    cache from before an engine refit/refreeze is rebuilt automatically,
    never served silently); otherwise builds it from
    `data_access.load_development()` (never `load_holdout_test`)."""
    if not force and not engine_versioning.is_stale(CACHE_PATH):
        df = pd.read_parquet(CACHE_PATH)
        df["time"] = pd.to_datetime(df["time"]); df["date"] = pd.to_datetime(df["date"])
        return df
    import data_access
    dev = data_access.load_development()
    sig = build_backtest_signals(dev)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sig.to_parquet(CACHE_PATH, index=False)
    engine_versioning.write_stamp(CACHE_PATH)
    return sig


if __name__ == "__main__":
    sig = load_or_build_signals(force=True)
    print(f"Built backtest signals: {len(sig)} rows, {sig['date'].nunique()} days -> {CACHE_PATH}")
    print(sig[["p_touch_5min", "p_touch_15min", "imbalance_reverting", "imbalance_accelerating"]].describe())
