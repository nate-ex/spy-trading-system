"""Shared live-market-state helpers -- LIVE STREAMING mode's imbalance
history/z-score bookkeeping and the touch/density engine live-prediction
bridges. Split out of `app.py` (unchanged behavior, just relocated) so a
second, separate dashboard (a personal trading desk, in its own private
repo) can reuse the exact same live-state logic without duplicating it --
see `app.py`'s import of this module for the research dashboard's own use.

`live_imbalance_velocity_acceleration` is new here (no prior live
equivalent existed): a live approximation of
`transition_state_features.py`'s `imbalance_velocity`/`imbalance_acceleration`
(first/second difference of `aggregate_dollar_imbalance` on a fixed
1-minute historical grid). The live version diffs against the most
recent PRIOR poll(s) instead, since live polls don't land on exact minute
boundaries -- an approximation of the historical definition, not identical
to it, called out explicitly rather than silently assumed equivalent.
"""

from __future__ import annotations

import json
from datetime import datetime, time
from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LIVE_IMBALANCE_HISTORY_CSV = REPO / "data" / "live_cache" / "live_imbalance_history.csv"
IMBALANCE_TOD_STD_BASELINE_PATH = REPO / "data" / "processed" / "imbalance_tod_std_baseline.json"


def append_live_imbalance_snapshot(now: datetime, agg_imbalance: float,
                                   path: Path = LIVE_IMBALANCE_HISTORY_CSV) -> None:
    """Append one (time, aggregate_dollar_imbalance) row to a small local
    cache file. TradeMas's `/imbalance` endpoint only returns a current
    snapshot, not a full-day history -- accumulating it ourselves, poll by
    poll, is what lets LIVE mode show the session-so-far trajectory (and,
    for a trading loop, compute velocity/acceleration). Persisted to disk
    (not `st.session_state`) because a Streamlit dashboard's auto-refresh
    does a full browser reload, which starts a fresh session each time;
    the same file also lets a non-Streamlit live trading process share the
    identical history without its own separate bookkeeping."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        if write_header:
            f.write("time,aggregate_dollar_imbalance\n")
        f.write(f"{now.isoformat()},{agg_imbalance}\n")


def load_live_imbalance_history(session_date, path: Path = LIVE_IMBALANCE_HISTORY_CSV) -> pd.DataFrame:
    """This session's accumulated (time, aggregate_dollar_imbalance) rows,
    filtered to `session_date` (a prior day's leftover cache shouldn't leak
    into today's chart/stats) and sorted chronologically."""
    cols = ["time", "aggregate_dollar_imbalance"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    # format="ISO8601": `append_live_imbalance_snapshot` writes
    # `now.isoformat()`, which OMITS the microsecond fraction whenever it's
    # exactly 0 (e.g. "...T10:41:00" vs "...T10:41:00.123456") -- plain
    # `pd.to_datetime` infers one strict format from the first rows and
    # then chokes the first time a row doesn't match it. ISO8601 mode
    # handles both forms per-row instead.
    df["time"] = pd.to_datetime(df["time"], format="ISO8601")
    df = df[df["time"].dt.date == session_date].sort_values("time").reset_index(drop=True)
    return df


def live_imbalance_zscore_intraday(current_agg: float, prior_hist: pd.DataFrame,
                                   min_periods: int = 3) -> float:
    """Same-day expanding z-score of the aggregate imbalance for LIVE mode --
    (current - mean of STRICTLY PRIOR polls today) / (population std of
    those prior polls), matching `state_features_common.
    expanding_intraday_zscore`'s definition exactly (the one
    `imbalance_zscore_intraday` uses historically/in REPLAY DEMO). `prior_hist`
    must be the poll history loaded BEFORE appending the current snapshot --
    the current reading is never part of its own reference stats.

    NOT the same thing as `live_feeds.imbalance_zscores` (a cross-sectional
    per-symbol ranking at one instant) -- averaging THAT across symbols is
    ~0 by construction (a set's z-scores always average to ~0).

    NaN until `min_periods` prior polls exist today, or if their std is 0."""
    if prior_hist.empty or len(prior_hist) < min_periods:
        return float("nan")
    prior_vals = prior_hist["aggregate_dollar_imbalance"].to_numpy(dtype=float)
    prior_std = float(prior_vals.std(ddof=0))
    if prior_std == 0:
        return float("nan")
    return float((current_agg - prior_vals.mean()) / prior_std)


@lru_cache(maxsize=1)
def load_imbalance_tod_std_baseline(path: Path = IMBALANCE_TOD_STD_BASELINE_PATH) -> dict:
    """The static `{"HH:MM": std}` table `live_imbalance_zscore_tod` reads
    -- see `relative_state_features.imbalance_tod_std_baseline`'s
    docstring for why this is a precomputed snapshot
    (`export_imbalance_tod_std_baseline.py`) rather than a live rolling
    computation. Cached for the process lifetime (one small JSON read,
    not per-poll) -- re-run the export script and restart the process to
    pick up a refreshed baseline, same convention as the frozen
    probability engines' JSONs."""
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _nearest_tod_std(baseline: dict, now_t: time) -> "float | None":
    """Nearest baseline clock-time at/before `now_t` (live polls don't
    land on exact grid minutes) -- falls back to the earliest available
    entry if `now_t` is before all of them, matching
    `_regular_day_grid`'s "no prior bar" convention."""
    if not baseline:
        return None
    now_str = now_t.strftime("%H:%M")
    candidates = sorted(k for k in baseline if k <= now_str)
    key = candidates[-1] if candidates else min(baseline)
    return baseline.get(key)


def live_imbalance_zscore_tod(current_agg: float, prior_hist: pd.DataFrame, now_t: time,
                              min_periods: int = 3,
                              baseline: "dict | None" = None) -> float:
    """Time-of-day-normalized live z-score: same-day expanding prior MEAN
    (identical formula to `live_imbalance_zscore_intraday`), but a
    precomputed cross-day time-of-day-matched STD
    (`load_imbalance_tod_std_baseline`) instead of today's own expanding
    std -- mirrors `relative_state_features.add_imbalance_zscore_tod`'s
    backtest definition (`imbalance_zscore_tod`). Fixes the same-day
    std's tendency to widen over the session if imbalance has real
    intraday heteroskedasticity, which otherwise silently raises the
    "significant" bar later in the day for reasons unrelated to whether
    the CURRENT reading is actually unusual right now.

    `baseline`: pass explicitly to avoid a repeated cached-load lookup in
    a hot loop; defaults to `load_imbalance_tod_std_baseline()`."""
    if prior_hist.empty or len(prior_hist) < min_periods:
        return float("nan")
    baseline = load_imbalance_tod_std_baseline() if baseline is None else baseline
    std = _nearest_tod_std(baseline, now_t)
    if std is None or std == 0:
        return float("nan")
    prior_vals = prior_hist["aggregate_dollar_imbalance"].to_numpy(dtype=float)
    return float((current_agg - prior_vals.mean()) / std)


def live_imbalance_velocity_acceleration(current_agg: float, prior_hist: pd.DataFrame) -> tuple[float, float]:
    """Live approximation of `transition_state_features.py`'s
    `imbalance_velocity` (= aggregate_dollar_imbalance[t] - [t-1], a
    1-minute first difference on the historical regular grid) and
    `imbalance_acceleration` (second difference). Live polls don't land on
    exact minute boundaries, so this diffs against the most recent one or
    two PRIOR polls instead of a strict 1-minute lag -- an approximation of
    the historical definition, not a reproduction of it. `prior_hist` must
    be the poll history loaded BEFORE appending the current snapshot (same
    convention as `live_imbalance_zscore_intraday`).

    Returns (nan, nan) with fewer than 2 prior polls (velocity needs 1
    prior point, acceleration needs 2)."""
    if prior_hist.empty:
        return float("nan"), float("nan")
    vals = prior_hist["aggregate_dollar_imbalance"].to_numpy(dtype=float)
    velocity = current_agg - vals[-1]
    if len(vals) < 2:
        return velocity, float("nan")
    prior_velocity = vals[-1] - vals[-2]
    acceleration = velocity - prior_velocity
    return velocity, acceleration


def touch_predictions(touch_engine, features: dict, now_t: time, tau_bps: float = 10.0) -> dict:
    """Per-horizon {P_touch, scale_bps, lognorm_s, lognorm_scale} from the
    TOUCH/first-passage engine -- drives the primary probability cards.
    One call works identically for REPLAY, LIVE dashboard, and a live
    trading loop: it only needs the same state-feature dict all three
    already produce (minutes_to_close, realized_vol_cum, realized_vol_30min,
    vix_close_lagged), not a precomputed grid column."""
    return touch_engine.predict(features, now=now_t, tau_bps=tau_bps)


def live_predictions(prediction: dict, engine_params: dict) -> dict:
    """Same shape as `app.py::replay_predictions`, from
    `live_stage1_feed.predict_live`'s output (already sigma_bps/eta/lambda/
    P_tail/P_upside/P_downside per horizon; mu_bps added from the frozen
    engine params)."""
    out = {}
    for h, d in prediction.items():
        mu_bps = float(engine_params["by_horizon"][h]["loc_mu"]) * 1e4
        out[h] = {**d, "mu_bps": mu_bps}
    return out
