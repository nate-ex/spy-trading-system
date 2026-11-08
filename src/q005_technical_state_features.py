"""
Feature construction for Q005 (SPY's own intraday technical state).
Zero new data -- everything derives from SPY's own OHLCV bars already in
`development.parquet`. Reuses `q003_persistence_features._regular_day_grid`
(already tested) for the per-day regular-grid reindexing rather than
duplicating that logic, and reuses `add_dynamics_features`'s
`today_return_so_far` / `minutes_to_close` directly rather than
recomputing them.

New features, all computed per day on the regular 1-min grid (13:00-16:15,
matching `q003_persistence_features.DAY_GRID_START/END`), using only data
up to and including each row's own time (no look-ahead):

- `today_return_so_far_sq`: square of the existing momentum feature
  (H-momentum-nonlinear's target).
- `realized_vol_cum`: sqrt(sum of squared 1-min log returns) from the
  day's first bar to the current row -- session-cumulative realized vol.
- `realized_vol_30min`: same, trailing 30-minute rolling window instead
  of session-cumulative (secondary/sensitivity variant).
- `vwap_deviation`: `(close - session_vwap) / session_vwap`, session VWAP
  = cumulative volume-weighted typical price `(high+low+close)/3` from
  the day's first bar to the current row.
- `range_used`: `(running_high - running_low) / session_open`, running
  high/low from the day's first bar to the current row.
- `volume_surprise`: cumulative volume so far today, relative to a
  20-trading-day trailing average of cumulative volume at the same
  time-of-day -- shift-then-rolling, identical convention to the existing
  ADV feature in `build_feature_table.py` (no look-ahead: only strictly
  prior days contribute to the trailing average).

Missing bars within a day (gaps in the SPY minute grid) are forward-filled
for price fields (close/high/low/open -- "last known price persists") and
zero-filled for volume (a gap contributes no incremental volume), mirroring
how `merge_asof` already treats gaps elsewhere in this project (explicit,
not silent).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from q003_persistence_features import _regular_day_grid, DAY_GRID_START, DAY_GRID_END

VOL_SURPRISE_WINDOW_DAYS = 20
REALIZED_VOL_TRAILING_MIN = 30


def _per_day_technical_features(day_df: pd.DataFrame, grid_start: str | None = None,
                                grid_end: str | None = None) -> pd.DataFrame:
    """`grid_start`/`grid_end` default to `DAY_GRID_START/END` (13:00-16:15)
    -- backtest callers (`build_technical_state_features`) never pass these,
    so the backtest grid is unchanged. The live trading desk's
    `live_stage1_feed.compute_live_features` passes a full-day override so
    the same feature math runs at any time of day, not just the 2-4pm
    trading window the strategies themselves are restricted to."""
    grid = _regular_day_grid(day_df, ["close", "high", "low", "open", "volume"],
                             grid_start=grid_start, grid_end=grid_end)

    # first_valid: the day's actual first bar. For the backtest's default
    # grid (13:00 start) this is always row 0, so everything below is a
    # no-op there. For the live desk's full-day grid override (see
    # `live_stage1_feed.LIVE_GRID_START/END`) the grid can start hours
    # before the session's first bar (pre-market/midnight rows) -- those
    # rows must read as "genuinely no data yet" (NaN), never silently
    # folded into the cumulative sums below as if they were zero-return
    # minutes, which would understate realized vol and let
    # realized_vol_30min warm up on padding instead of 30 real minutes.
    close_raw = grid["close"].to_numpy()
    has_data = ~np.isnan(close_raw)
    first_valid = int(np.argmax(has_data)) if has_data.any() else None

    close = grid["close"].ffill().to_numpy()
    high = grid["high"].ffill().to_numpy()
    low = grid["low"].ffill().to_numpy()
    volume = grid["volume"].fillna(0.0).to_numpy()
    n = len(grid)

    log_close = np.log(close)
    log_ret = np.diff(log_close, prepend=log_close[0])
    if first_valid is not None:
        log_ret[:first_valid] = np.nan  # no prior bar at all yet -- undefined, not zero
        log_ret[first_valid] = 0.0  # the day's actual first bar -- no prior bar WITHIN the day, zero not NaN/undefined

    sq_ret = log_ret ** 2  # NaN**2 stays NaN, propagates correctly below
    realized_vol_cum = np.full(n, np.nan)
    if first_valid is not None:
        realized_vol_cum[first_valid:] = np.sqrt(np.cumsum(sq_ret[first_valid:]))

    sq_ret_series = pd.Series(sq_ret)
    realized_vol_30min = np.sqrt(
        sq_ret_series.rolling(REALIZED_VOL_TRAILING_MIN, min_periods=REALIZED_VOL_TRAILING_MIN).sum().to_numpy()
    )

    typical_price = (high + low + close) / 3.0
    cum_vol = np.cumsum(volume)
    cum_vwap_num = np.cumsum(typical_price * volume)
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = cum_vwap_num / cum_vol
        vwap_deviation = (close - vwap) / vwap
    vwap_deviation[cum_vol == 0] = np.nan

    running_high = pd.Series(high).cummax().to_numpy()
    running_low = pd.Series(low).cummin().to_numpy()
    session_open = close[~np.isnan(close)][0] if (~np.isnan(close)).any() else np.nan
    range_used = (running_high - running_low) / session_open

    feat = pd.DataFrame(index=grid.index)
    feat["today_return_so_far_sq"] = ((close / session_open - 1) ** 2)
    feat["realized_vol_cum"] = realized_vol_cum
    feat["realized_vol_30min"] = realized_vol_30min
    feat["vwap_deviation"] = vwap_deviation
    feat["range_used"] = range_used
    feat["cum_volume"] = cum_vol
    feat = feat.reset_index().rename(columns={"index": "time"})
    return feat


def _add_volume_surprise(feat: pd.DataFrame) -> pd.DataFrame:
    """Shift-then-rolling trailing average of cum_volume at each
    time-of-day grid position, using only days strictly before the
    current one -- same no-look-ahead convention as the existing ADV
    feature."""
    feat = feat.copy()
    feat["grid_pos"] = feat.groupby("date").cumcount()
    wide = feat.pivot(index="date", columns="grid_pos", values="cum_volume").sort_index()
    expected = wide.shift(1).rolling(VOL_SURPRISE_WINDOW_DAYS, min_periods=10).mean()
    expected_long = expected.stack(future_stack=True).rename("expected_cum_volume").reset_index()
    feat = feat.merge(expected_long, on=["date", "grid_pos"], how="left")
    with np.errstate(invalid="ignore", divide="ignore"):
        feat["volume_surprise"] = feat["cum_volume"] / feat["expected_cum_volume"] - 1
    feat.loc[feat["expected_cum_volume"].isna() | (feat["expected_cum_volume"] == 0), "volume_surprise"] = np.nan
    return feat.drop(columns=["grid_pos", "cum_volume", "expected_cum_volume"])


def build_technical_state_features(dev_df: pd.DataFrame) -> pd.DataFrame:
    out_frames = []
    for date, day_df in dev_df.groupby("date", sort=True):
        f = _per_day_technical_features(day_df.sort_values("time"))
        f["date"] = date
        out_frames.append(f)
    feat = pd.concat(out_frames, ignore_index=True)
    return _add_volume_surprise(feat)


if __name__ == "__main__":
    import time
    from data_access import load_development
    from q003_persistence_features import add_dynamics_features

    t0 = time.time()
    dev = load_development()
    tech = build_technical_state_features(dev)
    dyn = add_dynamics_features(dev)[["time", "today_return_so_far", "minutes_to_close"]]
    merged = dev.merge(tech.drop(columns=["date"]), on="time", how="left").merge(dyn, on="time", how="left")
    print(f"Built Q005 feature table: {len(merged)} rows in {time.time()-t0:.1f}s")
    live = merged[~merged["imbalance_stale"]]
    print(live[["today_return_so_far", "today_return_so_far_sq", "realized_vol_cum", "realized_vol_30min",
                "vwap_deviation", "range_used", "volume_surprise"]].describe())
