"""
Ingest raw per-symbol NYSE imbalance files and build an aggregate signed
dollar-imbalance time series, entirely from data/raw/imbalances/ — the
Alerts feed is intentionally not used here (see questions/Q000, decision
logged 2026-07-15: Alerts are an estimate, not ground truth).

Aggregation approach: event-driven, not fixed-interval resampling. Each
symbol's imbalance state persists between its own ticks (a symbol that
last reported +50M buy imbalance at 2:01pm is still assumed +50M until
its next tick or the file's last tick that day). Every time *any* symbol
updates, we recompute the cross-sectional sum. This gives the true
step-function evolution of the aggregate signal without introducing
resampling artifacts — a fixed-grid view (e.g. for merging with 1-minute
SPY price data) can be derived from this via forward-fill, see
`resample_to_grid`.

Known exclusions applied here (see logs/research_log.md):
    - "Jan 2022.csv" dropped entirely (~96% bad timestamps).
    - "Feb 2022.csv" rows outside the sane trading window dropped
      (~1% of rows).
    - "August2024.csv" dropped entirely — confirmed byte-identical
      (same MD5, same #id sequence) to "August2025.csv"; contains zero
      genuine August 2024 data. Real August 2024 data is not currently
      available; this is a documented coverage gap, not a timestamp bug.
    - "November2024.csv" truncated to its genuine 2024-11-01 to
      2024-11-30 rows — the file is actually a bulk export running
      through 2025-04-24, and its December/January/February/March
      portions are exact row-for-row duplicates (verified via #id match)
      of the standalone December2024/January2025/February2025/March2025
      files, plus a partial duplicate of April2025. Keeping the full file
      would double/triple-count those months.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_validation import SANE_WINDOW_START, SANE_WINDOW_END

EXCLUDE_FILES_ENTIRELY = {
    "Jan 2022.csv",
    "August2024.csv",
}

# Files that are bulk multi-month exports overlapping with other
# standalone monthly files — truncate to the [start, end] (inclusive)
# window that is genuinely unique to this file; the rest duplicates
# other files in this directory (see module docstring / research_log.md).
TRUNCATE_FILE_WINDOWS = {
    "November2024.csv": ("2024-11-01", "2024-11-30"),
}


def load_raw_imbalance_dir(raw_dir: str | Path) -> pd.DataFrame:
    """
    Load and concatenate every CSV in `raw_dir`, applying the known
    exclusion rules. Returns a single dataframe with a `dollar_imbalance`
    column added (Imbalance * Side * LastSale — signed, positive = buy).
    """
    raw_dir = Path(raw_dir)
    frames = []
    for path in sorted(raw_dir.glob("*.csv")):
        if path.name in EXCLUDE_FILES_ENTIRELY:
            print(f"Skipping {path.name} (excluded — see research_log.md)")
            continue
        df = pd.read_csv(path)
        df["SaleTime"] = pd.to_datetime(
            df["SaleTime"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
        )
        n_bad = df["SaleTime"].isna().sum()
        if n_bad:
            print(f"  {path.name}: {n_bad} rows failed SaleTime parsing, dropped.")
            df = df.dropna(subset=["SaleTime"])

        if path.name in TRUNCATE_FILE_WINDOWS:
            win_start, win_end = TRUNCATE_FILE_WINDOWS[path.name]
            before = len(df)
            df = df[df["SaleTime"].between(win_start, win_end + " 23:59:59")]
            print(f"  {path.name}: truncated to {win_start}..{win_end} "
                  f"({before - len(df)} duplicate rows dropped, see research_log.md).")

        df["source_file"] = path.name
        frames.append(df)

    if not frames:
        raise ValueError(f"No usable files found in {raw_dir}")

    full = pd.concat(frames, ignore_index=True)

    # Filter out-of-window rows (handles Feb 2022's ~1% bad timestamps,
    # and acts as a general safety net for any other file).
    start = pd.to_datetime(SANE_WINDOW_START).time()
    end = pd.to_datetime(SANE_WINDOW_END).time()
    tod = full["SaleTime"].dt.time
    before = len(full)
    full = full[tod.between(start, end)].copy()
    dropped = before - len(full)
    if dropped:
        print(f"Dropped {dropped} rows ({100*dropped/before:.2f}%) outside "
              f"{SANE_WINDOW_START}-{SANE_WINDOW_END} sane window.")

    # Weekend safety net: 204 rows across Jan/Feb/May 2022 (all in the
    # already-flagged pre-production/unstable-collection era) carry
    # timestamps on Saturdays/Sundays, inside the sane time-of-day window
    # so the check above doesn't catch them — markets aren't open on
    # weekends, so these are unambiguously bad ticks (see research_log.md).
    before = len(full)
    full = full[full["SaleTime"].dt.dayofweek < 5].copy()
    dropped = before - len(full)
    if dropped:
        print(f"Dropped {dropped} rows falling on a weekend (not a trading day).")

    full["dollar_imbalance"] = full["Imbalance"] * full["Side"] * full["LastSale"]
    full = full.sort_values("SaleTime").reset_index(drop=True)
    return full


def build_aggregate_series(
    df: pd.DataFrame,
    value_col: str = "dollar_imbalance",
    output_col: str = "aggregate_dollar_imbalance",
) -> pd.DataFrame:
    """
    Event-driven cross-sectional sum of `value_col`, per trading day.
    Generalized so the same vetted vectorized logic aggregates either the
    dollar-imbalance signal or the ADV-normalized signal (see
    `imbalance_pct_adv` in build_feature_table.py) — not a replacement,
    both are meant to stay available side by side.

    Vectorized via cumsum-of-deltas: for each symbol, the change to the
    aggregate at each of its ticks is (new value - its previous value),
    with the first tick of the day contributing its full value (since it
    starts from an implicit zero). Summing these deltas in time order
    within each day reproduces the same running total as the naive
    "maintain a dict of last-known values" approach, without a Python
    row loop — this matters once we're past a few hundred thousand rows.

    Returns a dataframe with columns: date, time, `output_col`,
    n_symbols_reporting — one row per distinct tick timestamp within a day
    (ties at the same timestamp across symbols are merged into a single
    aggregate update, taking the state *after* all of that timestamp's
    updates have been applied). Rows with a null `value_col` (e.g. symbols
    with no ADV history yet, see build_feature_table.py) are dropped before
    aggregating — they contribute nothing rather than poisoning the sum
    with NaN.
    """
    df = df.copy()
    df["date"] = df["SaleTime"].dt.date
    df = df.dropna(subset=[value_col])
    df = df.sort_values(["date", "Symbol", "SaleTime"])

    delta = df.groupby(["date", "Symbol"])[value_col].diff()
    is_first_for_symbol = delta.isna()
    delta = delta.where(~is_first_for_symbol, df[value_col])
    df["delta"] = delta
    df["is_first_for_symbol"] = is_first_for_symbol.astype(int)

    # Re-sort into pure time order per day before cumsum — the groupby
    # above needed (date, Symbol, SaleTime) order to diff correctly, but
    # the running aggregate must accumulate in actual chronological order.
    df = df.sort_values(["date", "SaleTime"])
    df[output_col] = df.groupby("date")["delta"].cumsum()
    df["n_symbols_reporting"] = df.groupby("date")["is_first_for_symbol"].cumsum()

    agg = (
        df.groupby(["date", "SaleTime"], as_index=False)
        .agg(
            **{output_col: (output_col, "last")},
            n_symbols_reporting=("n_symbols_reporting", "last"),
        )
        .rename(columns={"SaleTime": "time"})
    )
    return agg


def resample_to_grid(
    agg_df: pd.DataFrame,
    freq: str = "1min",
    value_col: str = "aggregate_dollar_imbalance",
) -> pd.DataFrame:
    """
    Forward-fill the event-driven aggregate series onto a regular time
    grid per day — useful for merging with fixed-interval price data.
    `freq` follows pandas offset alias syntax (e.g. "1min", "30s").
    """
    out = []
    for date, day_df in agg_df.groupby("date", sort=True):
        day_df = day_df.set_index("time").sort_index()
        start = day_df.index.min().floor(freq)
        end = day_df.index.max().ceil(freq)
        grid = pd.date_range(start, end, freq=freq)
        gridded = day_df[value_col].reindex(
            day_df.index.union(grid)
        ).sort_index().ffill().reindex(grid)
        out.append(
            pd.DataFrame(
                {
                    "date": date,
                    "time": grid,
                    value_col: gridded.values,
                }
            )
        )
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    import sys

    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/imbalances"
    df = load_raw_imbalance_dir(raw_dir)
    print(f"Loaded {len(df)} rows across {df['source_file'].nunique()} files, "
          f"{df['date'].nunique() if 'date' in df else df['SaleTime'].dt.date.nunique()} trading days.")

    agg = build_aggregate_series(df)
    print(f"Built aggregate series: {len(agg)} distinct event timestamps "
          f"across {agg['date'].nunique()} days.")
    print(agg.groupby("date")["aggregate_dollar_imbalance"].agg(["min", "max"]).tail(10))
