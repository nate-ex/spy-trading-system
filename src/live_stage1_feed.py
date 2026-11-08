"""Live Stage 1 feature computation — MVI live probability dashboard.

Turns live SPY 1-minute bars AND the lagged VIX close into the exact same
Stage 1 Density Engine inputs (`minutes_to_close`, `realized_vol_cum`,
`realized_vol_30min`, `vix_close_lagged`) the frozen engine was fit on, by
reusing `q005_technical_state_features._per_day_technical_features`
DIRECTLY (not reimplementing it) — that function is the one place in this
codebase that defines `realized_vol_cum`/`realized_vol_30min`, so calling
it live is what guarantees the live path is mathematically identical to a
historical replay, not an approximation. See `validate_against_historical`
for the check.

SPY bars source priority (`fetch_today_spy_bars`'s `source="auto"`
default): **Bloomberg first if a Terminal session is available, else
Alpaca's free IEX feed** (`live_feeds.AlpacaBarFeed`) — no paid
subscription required for the Alpaca path, just a free account and
`ALPACA_API_KEY`/`ALPACA_SECRET_KEY` in `.env`. Massive is kept as a
further explicit fallback (`source="massive"`) but needs a PAID Massive
plan for intraday bars, so it isn't part of the automatic chain. VIX close
similarly prefers Bloomberg and falls back to the static `VIX_History.csv`
(Cboe) on any failure — see `load_vix_close_lagged`.

The Stage 1 Density Engine's mathematics (`stage1_density_engine.py`) are
NOT touched here — this module only prepares the state dict
`Stage1DensityEngine.predict()` already accepts. Zero retraining, zero
coefficient changes; only the feature *values* update live.

Setup (one-time, on the machine running the live dashboard):
  - Bloomberg path (optional, preferred when available): Terminal open and
    logged in, plus
        pip install blpapi --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/
        pip install xbbg
  - Alpaca path (free, no Terminal needed): sign up at
    https://alpaca.markets (paper account is enough for market data), put
    the key pair in `.env` (`alpaca-py` is already in `requirements.txt`).
"""

from __future__ import annotations

import sys
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE / "data"))

import q005_technical_state_features as q5tech  # noqa: E402
from stage1_density_engine import Stage1DensityEngine  # noqa: E402
from live_feeds import (  # noqa: E402
    EST, AlpacaBarFeed, BloombergBarFeed, MassiveSpyBarFeed, fetch_bloomberg_vix_close_lagged,
)

VIX_HISTORY_CSV = REPO / "data" / "raw" / "market_data" / "context" / "VIX_History.csv"
MARKET_CLOSE = time(16, 0)
REQUIRED_FEATURES = ["minutes_to_close", "realized_vol_cum", "realized_vol_30min", "vix_close_lagged"]

# Full-day override for compute_live_features' _per_day_technical_features
# call -- the backtest grid (q003_persistence_features.DAY_GRID_START/END,
# 13:00-16:15) stays the strategies' actual 2-4pm trading window untouched.
# This is only so the LIVE dashboard/desk can warm up and show a read at
# any time of day (e.g. mid-morning, ahead of the imbalance feed's
# afternoon activation), not a change to when trades are allowed.
LIVE_GRID_START = "00:00:00"
LIVE_GRID_END = "23:59:00"


def minutes_to_close(now: datetime, market_close: time = MARKET_CLOSE) -> float:
    """Minutes from `now` to today's 16:00 ET close — the same definition
    used throughout this codebase (see `time_state_features.py`)."""
    close_ts = datetime.combine(now.date(), market_close)
    return (close_ts - now).total_seconds() / 60.0


def _load_vix_close_lagged_from_csv(path: Path = VIX_HISTORY_CSV) -> float:
    """Static-file fallback: most recent available VIX close in
    `VIX_History.csv` (see `build_feature_table.load_vix`) -- NOT a live
    feed, may be stale. Only used if the live Bloomberg fetch fails."""
    vix = pd.read_csv(path)
    vix["date"] = pd.to_datetime(vix["DATE"], format="%m/%d/%Y")
    vix = vix.sort_values("date")
    return float(vix["CLOSE"].iloc[-1])


def load_vix_close_lagged(as_of=None, csv_path: Path = VIX_HISTORY_CSV, try_bloomberg: bool = True) -> float:
    """Lagged (most recently COMPLETED session's) VIX close. When
    `try_bloomberg` is True (default), sourced live from Bloomberg
    (`live_feeds.fetch_bloomberg_vix_close_lagged`) first, falling back to
    the static `VIX_History.csv` only if that fetch fails (e.g. Terminal
    not reachable) -- and says so loudly, rather than silently
    substituting a possibly-stale number. When `try_bloomberg` is False,
    Bloomberg is never even attempted -- goes straight to the static CSV.
    Set this to False for any caller that must never touch Bloomberg at
    all (e.g. `fetch_and_predict(..., source="alpaca")`, used by the
    personal trading dashboard, which has no business trying a Bloomberg
    Terminal connection on a machine that was never meant to have one)."""
    if not try_bloomberg:
        return _load_vix_close_lagged_from_csv(csv_path)
    try:
        return fetch_bloomberg_vix_close_lagged(as_of=as_of)
    except Exception as exc:
        print(f"[live_stage1_feed] Bloomberg VIX fetch failed ({exc}); "
              f"falling back to static {csv_path.name} (may be stale).")
        return _load_vix_close_lagged_from_csv(csv_path)


class NoLiveBarsYetError(ValueError):
    """Raised when there simply isn't a SPY bar to compute features from
    yet (before/at market open, or before `now`'s first bar posts) -- an
    EXPECTED, TRANSIENT state that self-heals once Bloomberg posts bars,
    not a configuration/connectivity problem. Kept as its own type (rather
    than a plain ValueError, like the genuine data-integrity error below)
    so the dashboard can show a calm "waiting for data" message instead of
    a "check your Bloomberg Terminal" error for this specific case."""


def compute_live_features(day_bars: pd.DataFrame, now: datetime, vix_close_lagged: float) -> dict:
    """Reuse `q005_technical_state_features._per_day_technical_features`
    unmodified on today's bars-so-far, take the state at/before `now`, and
    combine with `minutes_to_close`/`vix_close_lagged` — the exact
    BASE_SCALE_COLS the frozen Stage 1 engine expects. `realized_vol_30min`
    is left NaN (not fudged) before 30 minutes of bars exist, matching the
    historical warm-up behavior exactly."""
    if day_bars.empty:
        raise NoLiveBarsYetError("no SPY bars yet today -- can't compute live features")
    bar_dates = pd.to_datetime(day_bars["time"]).dt.date.unique()
    if len(bar_dates) > 1:
        # _regular_day_grid infers its whole 1-min grid from day_bars'
        # FIRST row's date alone -- a Bloomberg/xbbg response that mixes
        # two calendar dates (a normalize_bdib_frame timezone glitch, a
        # stale/duplicated poll, etc.) would otherwise silently compute
        # realized_vol_cum/realized_vol_30min against a grid missing most
        # of its own rows rather than failing loudly. Fail loudly instead
        # (a real data-integrity problem, NOT the expected/transient
        # "no bars yet" case above -- plain ValueError, not
        # NoLiveBarsYetError, so the dashboard treats it as a real error).
        raise ValueError(
            f"day_bars spans multiple calendar dates {sorted(bar_dates)} -- expected a "
            "single trading day's bars; refusing to compute features from mixed-day data "
            "(check the Bloomberg/xbbg response, see normalize_bdib_frame).")
    feat = q5tech._per_day_technical_features(
        day_bars.sort_values("time"), grid_start=LIVE_GRID_START, grid_end=LIVE_GRID_END
    ).set_index("time")
    idx = feat.index[feat.index <= now]
    if len(idx) == 0:
        raise NoLiveBarsYetError(f"no bar at/before {now} yet")
    row = feat.loc[idx[-1]]
    return {
        "time": idx[-1],
        "minutes_to_close": minutes_to_close(now),
        "realized_vol_cum": float(row["realized_vol_cum"]),
        "realized_vol_30min": float(row["realized_vol_30min"]) if pd.notna(row["realized_vol_30min"]) else float("nan"),
        "vix_close_lagged": vix_close_lagged,
    }


def is_warmed_up(features: dict) -> bool:
    """False while any required Stage 1 input is still NaN (e.g. the first
    30 minutes of a session, before realized_vol_30min exists)."""
    return all(pd.notna(features.get(k)) for k in REQUIRED_FEATURES)


def predict_live(engine: Stage1DensityEngine, features: dict, tau_bps: float = 10.0) -> dict:
    """Frozen-engine prediction from live features -- an identical call to
    the historical/replay path (`Stage1DensityEngine.predict`), zero
    retraining."""
    now_t = features["time"].time() if hasattr(features["time"], "time") else features["time"]
    state = {k: v for k, v in features.items() if k != "time"}
    return engine.predict(state, now=now_t, tau_bps=tau_bps)


def fetch_today_spy_bars(ticker: str = "SPY", source: str = "auto") -> tuple[pd.DataFrame, str]:
    """Today's 1-minute SPY bars so far, from `source`:
      - "auto" (default) — try Bloomberg (a local Terminal session via
        `xbbg`) first; if that raises ANY exception (not installed, no
        Terminal open, network hiccup, etc.), fall back to Alpaca's free
        IEX feed. This is the priority order requested for the live
        dashboard: Bloomberg when it's available, Alpaca otherwise --
        Alpaca needs no paid subscription, just `ALPACA_API_KEY`/
        `ALPACA_SECRET_KEY` in `.env`.
      - "bloomberg" — force Bloomberg only, no fallback (raises if
        unavailable).
      - "alpaca" — force Alpaca only.
      - "massive" — kept as a further fallback; requires a PAID Massive
        plan for intraday bars (the free tier is end-of-day only, see
        `live_feeds.py`'s module docstring), so not useful on a free-tier
        key.
    Returns `(bars, source_used)` so callers (the dashboard) can display
    which feed actually served the data -- important since "auto" may
    silently fall back."""
    if source == "auto":
        try:
            return BloombergBarFeed(tickers=[ticker]).fetch_today_bars(ticker), "bloomberg"
        except Exception as exc:
            print(f"[live_stage1_feed] Bloomberg SPY bar fetch failed ({exc}); "
                  f"falling back to Alpaca (free IEX feed).")
            return AlpacaBarFeed(tickers=[ticker]).fetch_today_bars(ticker), "alpaca"
    if source == "bloomberg":
        return BloombergBarFeed(tickers=[ticker]).fetch_today_bars(ticker), "bloomberg"
    if source == "alpaca":
        return AlpacaBarFeed(tickers=[ticker]).fetch_today_bars(ticker), "alpaca"
    if source == "massive":
        return MassiveSpyBarFeed(ticker=ticker).fetch_today_bars(), "massive"
    raise ValueError(f"unknown source {source!r} -- expected 'auto', 'bloomberg', 'alpaca', or 'massive'")


def fetch_and_predict(engine: Stage1DensityEngine, ticker: str = "SPY", tau_bps: float = 10.0,
                      source: str = "auto") -> dict:
    """One-shot: fetch today's SPY bars (Bloomberg first, Alpaca fallback
    by default -- see `fetch_today_spy_bars`), compute live features,
    predict with the frozen engine. Used directly by the Streamlit
    dashboard (re-run each poll; no asyncio needed for a request/response
    UI, unlike the async execution pipeline). `bars` is returned too
    (today's full bars-so-far, already fetched) so the dashboard can plot
    the SPY price trajectory without a second network call; `source_used`
    tells the dashboard which feed actually served this poll.

    The VIX close fetch also respects `source`: Bloomberg is only ever
    attempted when `source` is `"auto"` or `"bloomberg"` -- a caller that
    passes `source="alpaca"` (or `"massive"`) never touches Bloomberg for
    EITHER SPY bars or VIX, not just the bars half."""
    bars, source_used = fetch_today_spy_bars(ticker, source=source)
    # US/Eastern wall-clock, independent of this machine's own system
    # timezone setting -- plain datetime.now() would silently misalign
    # against the (now correctly Eastern-normalized) Bloomberg bar
    # timestamps if this machine's clock isn't set to US/Eastern, the same
    # class of bug normalize_bdib_frame just had to fix on the data side.
    now = datetime.now(EST).replace(tzinfo=None)
    vix = load_vix_close_lagged(as_of=now.date(), try_bloomberg=source in ("auto", "bloomberg"))
    features = compute_live_features(bars, now, vix)
    prediction = predict_live(engine, features, tau_bps=tau_bps) if is_warmed_up(features) else {}
    return {"features": features, "prediction": prediction, "warmed_up": is_warmed_up(features),
            "bars": bars, "source_used": source_used}


# --------------------------------------------------------------------------
# validation: live path must reproduce the historical path exactly
# --------------------------------------------------------------------------
def validate_against_historical(dev_df: pd.DataFrame, session_date, engine: Stage1DensityEngine) -> dict:
    """Recompute Stage 1 features for one already-known development-set day
    using ONLY that day's own price bars (as if replayed live, minute by
    minute through 14:00-16:00), and compare against the values already
    stored in `development.parquet` for the same rows. Confirms the live
    path is mathematically identical to the historical path, not an
    approximation -- development data only, holdout never touched here.
    """
    day = dev_df[pd.to_datetime(dev_df["date"]).dt.date == session_date].copy()
    if day.empty:
        raise ValueError(f"no development rows for {session_date}")
    day["time"] = pd.to_datetime(day["time"])
    bars = day[["time", "open", "high", "low", "close", "volume"]].sort_values("time")
    vix = float(day["vix_close_lagged"].iloc[0])

    win = day[(day["time"].dt.time >= time(14, 0)) & (day["time"].dt.time <= time(16, 0))]
    errs = {"realized_vol_cum": [], "realized_vol_30min": [], "minutes_to_close": []}
    checked = 0
    for _, hist_row in win.iterrows():
        now = pd.Timestamp(hist_row["time"]).to_pydatetime()
        try:
            live_feat = compute_live_features(bars, now, vix)
        except ValueError:
            continue
        errs["realized_vol_cum"].append(abs(live_feat["realized_vol_cum"] - hist_row["realized_vol_cum"]))
        errs["realized_vol_30min"].append(abs(live_feat["realized_vol_30min"] - hist_row["realized_vol_30min"]))
        errs["minutes_to_close"].append(abs(live_feat["minutes_to_close"] - hist_row["minutes_to_close"]))
        checked += 1
    return {"n_checked": checked,
            "max_abs_error": {k: (float(np.nanmax(v)) if v else None) for k, v in errs.items()}}


def main():
    import data_access

    dev = data_access.load_development()
    dev["time"] = pd.to_datetime(dev["time"])
    dev["date"] = pd.to_datetime(dev["date"])
    engine = Stage1DensityEngine.load()
    sample_date = pd.to_datetime(dev["date"]).dt.date.iloc[len(dev) // 2]
    report = validate_against_historical(dev, sample_date, engine)
    print(f"Validation vs. historical ({sample_date}): {report}")


if __name__ == "__main__":
    main()
