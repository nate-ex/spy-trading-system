"""
Feature construction for Q003 (imbalance persistence/state hypotheses).
Two categories, built differently:

1. **Breadth** (`build_breadth_series`) needs PER-SYMBOL state, which
   isn't carried into `data/processed/development.parquet` (only the
   already-aggregated columns are). Built from raw per-symbol ticks via
   `data/raw/imbalances/`, reusing `load_imbalances.build_aggregate_series`
   completely unchanged -- just with a 0/1 "is buy-side" indicator as the
   `value_col` instead of signed dollar amount. The exact same vetted,
   already-tested event-driven "current state persists between ticks"
   vectorized cumsum logic that produces a running dollar SUM produces a
   running symbol COUNT when fed a 0/1 indicator instead -- zero new
   aggregation logic, only a new input column.

2. **Duration since flip, slope, multi-window agreement, divergence**
   (`add_dynamics_features`) only need the aggregate's OWN history, which
   is already in `development.parquet` -- computed per day on a regular
   1-minute grid (reindexed from each day's rows, matching SPY's own
   1-min bar cadence) so "N minutes ago" is a well-defined lookup rather
   than an irregular-tick approximation. A lookback point that itself
   falls in the stale/pre-activation part of the day makes the dependent
   feature NaN for that row (not a live comparison), never silently
   filled.

Sign convention throughout: "sign" always refers to the AGGREGATE
dollar-imbalance's sign (positive = net buy), not any individual symbol's
side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from load_imbalances import load_raw_imbalance_dir, build_aggregate_series

RAW_IMBALANCES_DIR = "../data/raw/imbalances"

LOOKBACK_MINUTES = [5, 15, 30]  # H-agree's three windows; H-slope uses 5 (primary) and 15 (secondary)
DAY_GRID_START = "13:00:00"
DAY_GRID_END = "16:15:00"


def build_breadth_series(imb: pd.DataFrame) -> pd.DataFrame:
    """
    Event-driven breadth companion series: date, time,
    aggregate_dollar_imbalance, n_symbols_reporting, n_positive,
    n_negative, breadth (fraction of currently-reporting symbols sharing
    the aggregate's sign; NaN on the rare tick where the aggregate is
    exactly 0).
    """
    imb = imb.copy()
    imb["pos_indicator"] = (imb["Side"] == 1).astype(float)

    dollar_agg = build_aggregate_series(imb, value_col="dollar_imbalance", output_col="aggregate_dollar_imbalance")
    pos_agg = build_aggregate_series(imb, value_col="pos_indicator", output_col="n_positive")

    if len(dollar_agg) != len(pos_agg):
        raise ValueError(
            f"Event timestamp sets diverged between value_cols ({len(dollar_agg)} vs "
            f"{len(pos_agg)} rows) -- dollar_imbalance and pos_indicator should never "
            f"differ in null coverage since both derive from Side/Imbalance/LastSale, "
            f"all always-present raw fields. Investigate before trusting breadth."
        )

    merged = pd.merge(
        dollar_agg, pos_agg[["date", "time", "n_positive"]], on=["date", "time"], how="inner"
    )
    if len(merged) != len(dollar_agg):
        raise ValueError("Merge on (date, time) lost or duplicated rows -- event timestamps not aligned.")

    merged["n_negative"] = merged["n_symbols_reporting"] - merged["n_positive"]
    merged["breadth"] = np.select(
        [merged["aggregate_dollar_imbalance"] > 0, merged["aggregate_dollar_imbalance"] < 0],
        [merged["n_positive"] / merged["n_symbols_reporting"], merged["n_negative"] / merged["n_symbols_reporting"]],
        default=np.nan,
    )
    return merged


def merge_breadth_onto_dev(dev_df: pd.DataFrame, breadth_events: pd.DataFrame) -> pd.DataFrame:
    """
    merge_asof(direction="backward") of the event-driven breadth series
    onto development.parquet's SPY-minute grid, mirroring
    build_feature_table.py's own merge pattern exactly (including a
    breadth_stale flag analogous to imbalance_stale, since merge_asof
    silently carries a stale breadth value across any gap).
    """
    dev_sorted = dev_df.sort_values("time").copy()
    breadth_sorted = breadth_events.sort_values("time").copy()
    breadth_sorted["breadth_event_date"] = pd.to_datetime(breadth_sorted["date"])

    merged = pd.merge_asof(
        dev_sorted,
        breadth_sorted[["time", "breadth", "n_symbols_reporting", "n_positive", "n_negative", "breadth_event_date"]],
        on="time", direction="backward",
        suffixes=("", "_breadth"),
    )
    merged["date_ts"] = pd.to_datetime(merged["date"])
    merged["breadth_stale"] = merged["breadth_event_date"].dt.date != merged["date_ts"].dt.date
    return merged.drop(columns=["breadth_event_date", "date_ts"])


def _regular_day_grid(day_df: pd.DataFrame, value_cols: list,
                      grid_start: str | None = None, grid_end: str | None = None) -> pd.DataFrame:
    """Reindex one day's rows onto a strict 1-minute DatetimeIndex spanning
    the day's own grid, so N-minutes-ago lookups are well-defined.
    `grid_start`/`grid_end` default to the module's `DAY_GRID_START/END`
    (13:00-16:15, the imbalance-persistence backtest window) -- every
    existing caller (backtest feature-building) keeps that exact window
    untouched. Only the live trading desk's feature path
    (`live_stage1_feed.compute_live_features`) overrides these, since it
    needs a grid covering whenever the live loop happens to be polling,
    not just the 2-4pm window the strategies actually trade in."""
    day_df = day_df.set_index("time").sort_index()
    date_str = str(day_df.index[0].date())
    grid_start = grid_start or DAY_GRID_START
    grid_end = grid_end or DAY_GRID_END
    grid = pd.date_range(f"{date_str} {grid_start}", f"{date_str} {grid_end}", freq="1min")
    return day_df[value_cols].reindex(grid)


def add_dynamics_features(dev_df: pd.DataFrame, value_col: str = "aggregate_dollar_imbalance") -> pd.DataFrame:
    """
    Per day: sign, minutes_since_flip, slope_Nmin (N in LOOKBACK_MINUTES[:2]
    = [5, 15]), agree_Nmin (N in LOOKBACK_MINUTES = [5, 15, 30]),
    agreement_score (sum of the three agree flags), today_return_so_far
    (SPY's own return from the day's first available close price to the
    current row's close -- uses the FULL day's price history, not just
    live rows, since "the session so far" isn't gated by imbalance
    activation).

    A lookback comparison point that is itself stale/pre-activation
    (`imbalance_stale`) or simply outside the day's grid (first N minutes
    of the day) makes the dependent feature NaN for that row.
    """
    out_frames = []
    for date, day_df in dev_df.groupby("date", sort=True):
        day_df = day_df.sort_values("time")
        grid = _regular_day_grid(day_df, [value_col, "imbalance_stale", "close", "return_to_close"])

        val = grid[value_col].to_numpy()
        # Reindexing onto the full day grid introduces NaN for any minute
        # absent from the original rows (missing SPY bar, or before the
        # day's first bar) -- treat those as stale (no live reading),
        # never as a live False by accident of NaN-vs-bool coercion.
        stale = grid["imbalance_stale"].fillna(True).to_numpy().astype(bool)
        close = grid["close"].to_numpy()
        n = len(grid)

        sign = np.sign(val)
        sign[stale] = np.nan  # stale readings don't count as a live sign

        # minutes_since_flip: minutes since the live sign last differed
        # from the current live sign, restarting at each day's first live
        # reading. That first reading is flagged separately
        # (is_day_activation) -- it sets minutes_since_flip=0 by the same
        # "duration unknown before this, treat as fresh" convention as
        # Q000/Q001, but it is NOT a genuine reversal (there's no prior
        # live state that day to reverse from) and must be excluded from
        # any flip-artifact sanity check, which is specifically about
        # whether genuine intraday reversals are real or latency noise.
        minutes_since_flip = np.full(n, np.nan)
        is_day_activation = np.zeros(n, dtype=bool)
        last_flip_idx = None
        prev_sign = None
        for i in range(n):
            if stale[i] or np.isnan(sign[i]):
                continue
            if prev_sign is None:
                is_day_activation[i] = True
                last_flip_idx = i
            elif sign[i] != prev_sign:
                last_flip_idx = i
            prev_sign = sign[i]
            minutes_since_flip[i] = i - last_flip_idx

        feat = pd.DataFrame(index=grid.index)
        feat["sign"] = sign
        feat["minutes_since_flip"] = minutes_since_flip
        feat["is_day_activation"] = is_day_activation
        feat["is_genuine_flip"] = (minutes_since_flip == 0) & (~is_day_activation)

        for N in LOOKBACK_MINUTES[:2]:  # slope: 5, 15
            lag_val = pd.Series(val).shift(N).to_numpy()
            lag_stale = pd.Series(stale.astype(float)).shift(N).to_numpy()
            slope = (val - lag_val) / N
            slope[(lag_stale != 0) | np.isnan(lag_stale) | stale] = np.nan
            feat[f"slope_{N}min"] = slope

        agree_cols = []
        for N in LOOKBACK_MINUTES:  # agreement: 5, 15, 30
            lag_sign = pd.Series(sign).shift(N).to_numpy()
            agree = (sign == lag_sign).astype(float)
            agree[np.isnan(sign) | np.isnan(lag_sign)] = np.nan
            feat[f"agree_{N}min"] = agree
            agree_cols.append(f"agree_{N}min")
        feat["agreement_score"] = feat[agree_cols].sum(axis=1, skipna=False)

        session_open_close = close[~np.isnan(close)][0] if (~np.isnan(close)).any() else np.nan
        feat["today_return_so_far"] = close / session_open_close - 1

        # Matches Q001's label_end_time convention exactly (date + 16:00:00,
        # the modal close-reference bar) for consistency across questions.
        close_ts = pd.Timestamp(f"{date} 16:00:00")
        feat["minutes_to_close"] = (close_ts - grid.index).total_seconds() / 60.0

        feat["date"] = date
        feat = feat.reset_index().rename(columns={"index": "time"})
        out_frames.append(feat)

    return pd.concat(out_frames, ignore_index=True)


def build_q003_feature_table(raw_imbalances_dir: str = RAW_IMBALANCES_DIR) -> pd.DataFrame:
    """
    Full pipeline: dev data + dynamics features (from development.parquet
    alone) + breadth (from raw per-symbol ticks), merged into one table
    at the SPY-minute grid. Entry point for q003_analysis.py.
    """
    from data_access import load_development

    dev = load_development()
    dynamics = add_dynamics_features(dev)
    dev_with_dynamics = dev.merge(
        dynamics.drop(columns=["date"]), on="time", how="left"
    )

    imb = load_raw_imbalance_dir(raw_imbalances_dir)
    breadth_events = build_breadth_series(imb)
    full = merge_breadth_onto_dev(dev_with_dynamics, breadth_events)
    return full


if __name__ == "__main__":
    import time

    t0 = time.time()
    table = build_q003_feature_table()
    print(f"Built Q003 feature table: {len(table)} rows in {time.time()-t0:.1f}s")
    print(table.columns.tolist())
    live = table[(~table["imbalance_stale"]) & (~table["breadth_stale"])]
    print(f"Rows live on both imbalance and breadth: {len(live)}")
    print(live[["sign", "minutes_since_flip", "slope_5min", "slope_15min",
                "agreement_score", "breadth", "today_return_so_far"]].describe())
