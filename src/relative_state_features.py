"""
Relative State: how UNUSUAL is the current value of another family's
state variable, relative to its own recent history -- not the raw value
itself. Three related computations, all from `state_features_common`:
    (a) same-day expanding z-score (`expanding_intraday_zscore`)
    (b) multi-day, same-time-of-day trailing percentile
        (`trailing_percentile_same_time_of_day`)
    (c) `imbalance_zscore_tod` -- (a)'s same-day expanding MEAN, but a
        cross-day, time-of-day-matched trailing STD
        (`trailing_std_same_time_of_day`) instead of (a)'s same-day-only
        std. (a)'s denominator keeps widening across the session if
        imbalance has real intraday heteroskedasticity (e.g. builds
        toward the close), which silently raises the "significant" bar
        later in the day for reasons unrelated to whether the CURRENT
        reading is actually unusual right now -- this is an alternative
        that isolates that effect. Kept alongside (a), not replacing it:
        `imbalance_zscore_intraday` is what the existing frozen engines/
        strategies were built and validated against.

Every source column consumed here (`aggregate_dollar_imbalance`,
`realized_vol_cum`, `cum_volume_today`, `range_used`, `breadth`,
`trailing_return_5min`, `vix_close_lagged`) must already be present --
this module runs after Q003/Q005/Q006 and the new order-flow/SPY/
volatility/market-context modules are merged in.

Imbalance- and breadth-derived inputs are explicitly masked by their own
staleness flags (`imbalance_stale`, `breadth_stale`) before entering
either the z-score or the percentile computation -- a carried-forward
stale value is not today's real reading and must not contaminate the
historical reference distribution or receive a percentile of its own,
same discipline `q003_persistence_features.py` already applies. SPY-
price-derived inputs have no such staleness concept (SPY's own price is
always live during trading hours), so no masking is needed for those.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from q003_persistence_features import _regular_day_grid
from state_features_common import (
    trailing_percentile_same_time_of_day, expanding_intraday_zscore, trailing_std_same_time_of_day,
    trailing_zscore_same_time_of_day,
)

IMBALANCE_ZSCORE_MIN_PERIODS = 3   # matches expanding_intraday_zscore's own default

VIX_ZSCORE_WINDOW_DAYS = 20
VIX_PERCENTILE_WINDOW_DAYS = 60
VIX_PERCENTILE_MIN_PERIODS = 10

PERCENTILE_SOURCE_COLS = {
    "aggregate_dollar_imbalance": "imbalance_relative_to_history",
    "realized_vol_cum": "realized_vol_relative_to_history",
    "cum_volume_today": "volume_relative_to_history",
    "range_used": "range_relative_to_history",
    "breadth": "breadth_relative_to_history",
    "trailing_return_5min": "momentum_relative_to_history",
}


def add_imbalance_zscore_tod(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `imbalance_zscore_tod`: (a)'s same-day expanding prior MEAN
    formula, but a cross-day, time-of-day-matched trailing STD
    (`trailing_std_same_time_of_day`) instead of (a)'s same-day-only std.
    Self-contained (does its own masking + regridding of
    `aggregate_dollar_imbalance` only) so it can run standalone -- e.g. to
    backfill this one column onto an already-built feature table/
    development set without re-running the rest of
    `add_relative_state_features` (which would duplicate-merge every
    other column already present)."""
    df = df.copy()
    imbalance_masked = df["aggregate_dollar_imbalance"].where(~df["imbalance_stale"].fillna(True))

    grid_source = df[["date", "time"]].copy()
    grid_source["aggregate_dollar_imbalance"] = imbalance_masked
    grid_frames = []
    for date, day_df in grid_source.groupby("date", sort=True):
        day_df = day_df.sort_values("time")
        grid = _regular_day_grid(day_df, ["aggregate_dollar_imbalance"])
        grid = grid.reset_index().rename(columns={"index": "time"})
        grid["date"] = date
        grid_frames.append(grid)
    grid_all = pd.concat(grid_frames, ignore_index=True)

    grid_all["_imbalance_prior_mean"] = grid_all.groupby("date")["aggregate_dollar_imbalance"].transform(
        lambda s: s.shift(1).expanding(min_periods=IMBALANCE_ZSCORE_MIN_PERIODS).mean()
    )
    grid_all = trailing_std_same_time_of_day(
        grid_all, "aggregate_dollar_imbalance", output_col="_imbalance_tod_std"
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        grid_all["imbalance_zscore_tod"] = (
            (grid_all["aggregate_dollar_imbalance"] - grid_all["_imbalance_prior_mean"])
            / grid_all["_imbalance_tod_std"]
        )
    grid_all.loc[
        (grid_all["_imbalance_tod_std"] == 0) | grid_all["_imbalance_tod_std"].isna(),
        "imbalance_zscore_tod",
    ] = np.nan

    return df.merge(grid_all[["time", "imbalance_zscore_tod"]], on="time", how="left")


def add_imbalance_price_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `imbalance_price_spread` -- a genuine, comparable-scale spread
    between the imbalance level and SPY's own cumulative session return,
    for testing the "does the SPY-vs-imbalance chart's visual mean
    reversion hold up" hypothesis (spurious dual-independent-axis
    autoscaling was the leading suspect -- see the conversation that
    prompted this).

    `imbalance_zscore_crossday`: `aggregate_dollar_imbalance` (staleness-
    masked), z-scored via `trailing_zscore_same_time_of_day` -- BOTH mean
    and std from the trailing cross-day window at this clock position.
    `price_return_zscore_crossday`: `today_return_so_far` (SPY's own
    cumulative return since the session open -- no staleness concept, see
    module docstring), z-scored the exact same way.

    Deliberately NOT `imbalance_zscore_tod` (which mixes a same-day-
    expanding mean with a cross-day std) -- subtracting two z-scores only
    means something if both were calibrated identically. Using
    `trailing_zscore_same_time_of_day` for both sides is what makes
    `imbalance_price_spread = imbalance_zscore_crossday -
    price_return_zscore_crossday` an honest "is imbalance unusually far
    ahead of/behind SPY's own move, relative to how far ahead/behind it
    typically runs at this time of day" statistic, not two arbitrarily-
    scaled numbers subtracted from each other.

    Self-contained (own masking/regridding), like `add_imbalance_zscore_tod`
    -- runs standalone against an already-built development set.
    """
    df = df.copy()
    imbalance_masked = df["aggregate_dollar_imbalance"].where(~df["imbalance_stale"].fillna(True))

    grid_source = df[["date", "time", "today_return_so_far"]].copy()
    grid_source["aggregate_dollar_imbalance"] = imbalance_masked
    grid_frames = []
    for date, day_df in grid_source.groupby("date", sort=True):
        day_df = day_df.sort_values("time")
        grid = _regular_day_grid(day_df, ["aggregate_dollar_imbalance", "today_return_so_far"])
        grid = grid.reset_index().rename(columns={"index": "time"})
        grid["date"] = date
        grid_frames.append(grid)
    grid_all = pd.concat(grid_frames, ignore_index=True)

    grid_all = trailing_zscore_same_time_of_day(
        grid_all, "aggregate_dollar_imbalance", output_col="imbalance_zscore_crossday"
    )
    grid_all = trailing_zscore_same_time_of_day(
        grid_all, "today_return_so_far", output_col="price_return_zscore_crossday"
    )
    grid_all["imbalance_price_spread"] = (
        grid_all["imbalance_zscore_crossday"] - grid_all["price_return_zscore_crossday"]
    )

    keep_cols = ["time", "imbalance_zscore_crossday", "price_return_zscore_crossday", "imbalance_price_spread"]
    return df.merge(grid_all[keep_cols], on="time", how="left")


def imbalance_tod_std_baseline(df: pd.DataFrame) -> dict[str, float]:
    """Static `{"HH:MM": std}` lookup for LIVE use (`live_signal_state.
    live_imbalance_zscore_tod`) -- a live poll can't do the multi-day
    pandas rolling `trailing_std_same_time_of_day` does on the fly, so
    this exports the MOST RECENT available estimate (the trailing-window
    std as of the last date in `df`) as a small static table instead.
    Same masking/regridding as `add_imbalance_zscore_tod`; only the
    output shape differs (one row per clock-time, not one row per
    (date, clock-time)).

    Regenerate this (`export_imbalance_tod_std_baseline.py`) whenever
    `development.parquet`/`feature_table.parquet` is rebuilt -- it's a
    point-in-time snapshot, not something that updates itself."""
    imbalance_masked = df["aggregate_dollar_imbalance"].where(~df["imbalance_stale"].fillna(True))
    grid_source = df[["date", "time"]].copy()
    grid_source["aggregate_dollar_imbalance"] = imbalance_masked
    grid_frames = []
    for date, day_df in grid_source.groupby("date", sort=True):
        day_df = day_df.sort_values("time")
        grid = _regular_day_grid(day_df, ["aggregate_dollar_imbalance"])
        grid = grid.reset_index().rename(columns={"index": "time"})
        grid["date"] = date
        grid_frames.append(grid)
    grid_all = pd.concat(grid_frames, ignore_index=True)
    grid_all = trailing_std_same_time_of_day(grid_all, "aggregate_dollar_imbalance", output_col="_std")

    last_date = grid_all["date"].max()
    last_day = grid_all[grid_all["date"] == last_date].sort_values("time")
    return {
        pd.Timestamp(t).strftime("%H:%M"): (None if pd.isna(v) else float(v))
        for t, v in zip(last_day["time"], last_day["_std"])
    }


def _vix_percentile_60d(vix_by_date: pd.DataFrame) -> pd.Series:
    def _pct_of_last(window: np.ndarray) -> float:
        *hist, current = window
        hist = np.asarray(hist, dtype=float)
        hist = hist[~np.isnan(hist)]
        if len(hist) < VIX_PERCENTILE_MIN_PERIODS or np.isnan(current):
            return np.nan
        return float((hist < current).mean())

    return vix_by_date["vix_close_lagged"].rolling(
        VIX_PERCENTILE_WINDOW_DAYS + 1, min_periods=VIX_PERCENTILE_MIN_PERIODS + 1
    ).apply(_pct_of_last, raw=True)


def add_relative_state_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    imbalance_masked = df["aggregate_dollar_imbalance"].where(~df["imbalance_stale"].fillna(True))
    breadth_masked = df["breadth"].where(~df["breadth_stale"].fillna(True))

    # (a) same-day expanding z-score of the (staleness-masked) imbalance level.
    z_frames = []
    tmp = df[["date", "time"]].copy()
    tmp["_imbalance_masked"] = imbalance_masked
    for date, day_df in tmp.groupby("date", sort=True):
        day_df = day_df.sort_values("time")
        z = expanding_intraday_zscore(day_df, "_imbalance_masked")
        z_frames.append(pd.DataFrame({"time": day_df["time"], "imbalance_zscore_intraday": z}))
    df = df.merge(pd.concat(z_frames, ignore_index=True), on="time", how="left")

    # (b) multi-day, same-time-of-day percentile rank, on several columns.
    grid_source = df.copy()
    grid_source["aggregate_dollar_imbalance"] = imbalance_masked
    grid_source["breadth"] = breadth_masked

    grid_cols = list(PERCENTILE_SOURCE_COLS.keys())
    grid_frames = []
    for date, day_df in grid_source.groupby("date", sort=True):
        day_df = day_df.sort_values("time")
        grid = _regular_day_grid(day_df, grid_cols)
        grid = grid.reset_index().rename(columns={"index": "time"})
        grid["date"] = date
        grid_frames.append(grid)
    grid_all = pd.concat(grid_frames, ignore_index=True)

    for col, out_col in PERCENTILE_SOURCE_COLS.items():
        grid_all = trailing_percentile_same_time_of_day(grid_all, col, output_col=out_col)

    keep_cols = ["time"] + list(PERCENTILE_SOURCE_COLS.values())
    df = df.merge(grid_all[keep_cols], on="time", how="left")

    # (c) imbalance_zscore_tod -- see add_imbalance_zscore_tod's docstring.
    df = add_imbalance_zscore_tod(df)

    # VIX day-level relative stats.
    vix_by_date = (
        df[["date", "vix_close_lagged"]]
        .drop_duplicates(subset="date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    roll_mean = vix_by_date["vix_close_lagged"].rolling(
        VIX_ZSCORE_WINDOW_DAYS, min_periods=VIX_ZSCORE_WINDOW_DAYS
    ).mean()
    roll_std = vix_by_date["vix_close_lagged"].rolling(
        VIX_ZSCORE_WINDOW_DAYS, min_periods=VIX_ZSCORE_WINDOW_DAYS
    ).std(ddof=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        vix_by_date["vix_zscore_20d"] = (vix_by_date["vix_close_lagged"] - roll_mean) / roll_std
    vix_by_date.loc[(roll_std == 0) | roll_std.isna(), "vix_zscore_20d"] = np.nan
    vix_by_date["vix_trend"] = vix_by_date["vix_close_lagged"] - roll_mean
    vix_by_date["vix_percentile_60d"] = _vix_percentile_60d(vix_by_date)

    df = df.merge(
        vix_by_date[["date", "vix_zscore_20d", "vix_trend", "vix_percentile_60d"]],
        on="date", how="left",
    )

    return df
