"""
Shared helpers for the Relative State family (and anything else needing
the same pattern) -- generalizes q005_technical_state_features.py's
already-tested `_add_volume_surprise` shift-then-rolling logic (only
strictly-prior days contribute, zero look-ahead) from one hardcoded
ratio-to-mean into a reusable percentile helper, plus a same-day
expanding z-score for the "relative to *today's own* history" sense.

Four related helpers:
    1. `trailing_percentile_same_time_of_day` -- multi-day: is this value
       unusual for this time of day, compared to the trailing N prior
       trading days at the same time-of-day grid position.
    2. `expanding_intraday_zscore` -- same-day only: is this value unusual
       relative to *today's own* readings so far.
    3. `trailing_std_same_time_of_day` -- a cross-day, time-of-day-matched
       std, meant as an alternative denominator for (2): (2)'s same-day
       std keeps widening over the session if the signal has intraday
       heteroskedasticity, silently raising the "significant" bar later
       in the day. See `relative_state_features.py`'s
       `imbalance_zscore_tod` for the combined result (same-day mean
       from (2)'s formula, cross-day std from this).
    4. `trailing_zscore_same_time_of_day` -- like (3), but BOTH mean and
       std come from the trailing cross-day window (no same-day-expanding
       mean mixed in). Two series normalized this exact same way are on a
       genuinely comparable scale -- built for `imbalance_price_spread`
       (see `relative_state_features.py`), which subtracts one such
       z-score from another and needs both sides calibrated identically,
       not one same-day-hybrid and one pure-cross-day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WINDOW_DAYS = 20
DEFAULT_MIN_PRIOR_DAYS = 10


def trailing_percentile_same_time_of_day(
    feat_df: pd.DataFrame,
    value_col: str,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_periods: int = DEFAULT_MIN_PRIOR_DAYS,
    output_col: str | None = None,
) -> pd.DataFrame:
    """
    For each row, the percentile rank (0-1) of `value_col` at this row's
    time-of-day grid position, relative to the trailing `window_days`
    STRICTLY PRIOR days' values at that same grid position. Requires one
    row per (date, time-of-day grid position) -- i.e. already on a
    regular per-day grid (see q003_persistence_features._regular_day_grid).
    NaN wherever fewer than `min_periods` valid prior-day observations
    exist at that grid position, or the current value itself is NaN.
    """
    output_col = output_col or f"{value_col}_percentile"
    feat = feat_df.copy()
    feat["_grid_pos"] = feat.groupby("date").cumcount()
    wide = feat.pivot(index="date", columns="_grid_pos", values=value_col).sort_index()

    def _pct_of_last(window: np.ndarray) -> float:
        *hist, current = window
        hist = np.asarray(hist, dtype=float)
        hist = hist[~np.isnan(hist)]
        if len(hist) < min_periods or np.isnan(current):
            return np.nan
        return float((hist < current).mean())

    pct = wide.rolling(window_days + 1, min_periods=min_periods + 1).apply(_pct_of_last, raw=True)
    pct_long = pct.stack(future_stack=True).rename(output_col).reset_index()
    feat = feat.merge(pct_long, on=["date", "_grid_pos"], how="left")
    return feat.drop(columns=["_grid_pos"])


def trailing_std_same_time_of_day(
    feat_df: pd.DataFrame,
    value_col: str,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_periods: int = DEFAULT_MIN_PRIOR_DAYS,
    output_col: str | None = None,
) -> pd.DataFrame:
    """
    For each row, the population std (ddof=0) of `value_col` at this row's
    time-of-day grid position, across the trailing `window_days` STRICTLY
    PRIOR days at that same grid position (current day fully excluded via
    `shift(1)` -- same no-look-ahead convention as
    `trailing_percentile_same_time_of_day`/
    `q005_technical_state_features._add_volume_surprise`). Requires one row
    per (date, time-of-day grid position) -- i.e. already on a regular
    per-day grid (see q003_persistence_features._regular_day_grid). NaN
    wherever fewer than `min_periods` valid prior-day observations exist
    at that grid position.

    Built as a cross-day, time-of-day-matched alternative denominator for
    `expanding_intraday_zscore`'s same-day-only std: if the underlying
    signal has genuine intraday heteroskedasticity (e.g. imbalance
    building toward the close), a same-day expanding std keeps growing
    over the session, silently raising the z-score bar for "significant"
    later in the day for reasons unrelated to whether the CURRENT reading
    is actually unusual right now. This std is instead computed from the
    SAME time-of-day position on other days, so it doesn't drift within
    a session.
    """
    output_col = output_col or f"{value_col}_tod_std"
    feat = feat_df.copy()
    feat["_grid_pos"] = feat.groupby("date").cumcount()
    wide = feat.pivot(index="date", columns="_grid_pos", values=value_col).sort_index()
    std = wide.shift(1).rolling(window_days, min_periods=min_periods).std(ddof=0)
    std_long = std.stack(future_stack=True).rename(output_col).reset_index()
    feat = feat.merge(std_long, on=["date", "_grid_pos"], how="left")
    return feat.drop(columns=["_grid_pos"])


def trailing_zscore_same_time_of_day(
    feat_df: pd.DataFrame,
    value_col: str,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_periods: int = DEFAULT_MIN_PRIOR_DAYS,
    output_col: str | None = None,
) -> pd.DataFrame:
    """
    For each row, `value_col`'s z-score against the trailing `window_days`
    STRICTLY PRIOR days at that same time-of-day grid position -- BOTH the
    mean and the std come from that same cross-day trailing window (unlike
    `expanding_intraday_zscore`, whose mean is same-day-expanding). Same
    no-look-ahead convention as the other helpers here (`shift(1)` before
    rolling -- current day never contributes to its own reference stats).
    Requires one row per (date, time-of-day grid position) -- i.e. already
    on a regular per-day grid (see q003_persistence_features.
    _regular_day_grid). NaN wherever fewer than `min_periods` valid
    prior-day observations exist at that grid position, or the trailing
    std is exactly 0.
    """
    output_col = output_col or f"{value_col}_zscore_crossday"
    feat = feat_df.copy()
    feat["_grid_pos"] = feat.groupby("date").cumcount()
    wide = feat.pivot(index="date", columns="_grid_pos", values=value_col).sort_index()
    prior = wide.shift(1)
    mean = prior.rolling(window_days, min_periods=min_periods).mean()
    std = prior.rolling(window_days, min_periods=min_periods).std(ddof=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (wide - mean) / std
    z = z.where(std != 0)
    z_long = z.stack(future_stack=True).rename(output_col).reset_index()
    feat = feat.merge(z_long, on=["date", "_grid_pos"], how="left")
    return feat.drop(columns=["_grid_pos"])


def expanding_intraday_zscore(day_df: pd.DataFrame, value_col: str, min_periods: int = 3) -> np.ndarray:
    """
    Same-day-only expanding z-score of `value_col`: (current - mean of
    STRICTLY PRIOR same-day readings) / (std of strictly prior same-day
    readings), population std (ddof=0). Excludes the current reading from
    its own reference stats (the "surprise" framing -- a reading is
    compared to what was already known before it arrived, not to a
    mean/std that includes itself). NaN until `min_periods` prior live
    readings exist that day, and wherever the prior std is exactly 0.
    """
    vals = day_df[value_col].to_numpy(dtype=float)
    s = pd.Series(vals)
    prior = s.shift(1)
    prior_mean = prior.expanding(min_periods=min_periods).mean()
    prior_std = prior.expanding(min_periods=min_periods).std(ddof=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (s - prior_mean) / prior_std
    z[(prior_std == 0) | prior_std.isna()] = np.nan
    return z.to_numpy()
