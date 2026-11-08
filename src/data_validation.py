"""
Validation for raw NYSE imbalance data files, matching the real schema
confirmed from sample files (2022 + Jan-Mar 2026):

Imbalances files — one row per (Symbol, message) tick:
    #id           : int, message sequence id
    Symbol        : str, NYSE-listed ticker
    Side          : {1, -1}   (1 = buy imbalance, -1 = sell imbalance)
    Imbalance     : int, imbalance shares
    PairOff       : int, paired/matched shares at this timestamp
    LastSale      : float, last trade price
    SaleTime      : timestamp string, "M/D/YYYY H:MM:SS AM/PM"
    FlipOn        : {0, 1} (1 if Side changed vs. this ticker's prior tick)

Alerts files — a state-log of the aggregate (all-symbol) imbalance level,
one row per level change:
    #id           : int
    Side          : "Buy" / "Sell" (string, not 1/-1 — different from
                    Imbalances' Side encoding, don't conflate the two)
    Amount        : float, aggregate imbalance in dollars, rounded to the
                    nearest $1M — NOT a fixed threshold bucket (confirmed:
                    values like 407,000,000 and 833,000,000 appear, so
                    this is a real running total, just coarsely rounded)
    EntryDate     : timestamp, when this level began
    EndDate       : timestamp, when this level changed (i.e. the next
                    row's EntryDate, approximately)

Known data quality issues found in the sample files (see research_log.md):
    - "Jan 2022.csv" is ~96% timestamps outside any sane trading window —
      looks like early/test data before the collection system stabilized.
      Treat as unusable until confirmed otherwise.
    - "Feb 2022.csv" has ~1% of rows outside a sane trading window —
      small enough to filter row-by-row rather than discard the month.
    - Alerts are only provided from Jan 2026 onward in the sample; no
      Alerts history for 2022-2025 has been provided yet.
"""

from __future__ import annotations

import pandas as pd

IMBALANCE_EXPECTED_COLUMNS = {
    "#id": "int64",
    "Symbol": "object",
    "Side": "int64",
    "Imbalance": "int64",
    "PairOff": "int64",
    "LastSale": "float64",
    "SaleTime": "datetime64[ns]",
    "FlipOn": "int64",
}

ALERTS_EXPECTED_COLUMNS = {
    "#id": "int64",
    "Side": "object",  # "Buy" / "Sell" — string, unlike Imbalances.Side
    "Amount": "float64",
    "EntryDate": "datetime64[ns]",
    "EndDate": "datetime64[ns]",
}

# Sample data showed EntryDate as early as 1:29 PM on some days, not a
# clean 2:00 PM start — keep this loose and let a WARNING (not an ERROR)
# flag genuinely odd timestamps, rather than hard-coding an assumption
# about the window that the data itself doesn't support.
SANE_WINDOW_START = "13:00"
SANE_WINDOW_END = "16:15"


class ValidationReport:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def ok(self) -> bool:
        return len(self.errors) == 0

    def __repr__(self) -> str:
        lines = [f"ValidationReport(ok={self.ok()})"]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def _check_time_window(df: pd.DataFrame, time_col: str, report: ValidationReport) -> None:
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        report.warnings.append(
            f"`{time_col}` is not parsed as datetime — parse before validation "
            "for the time-window check to run."
        )
        return
    times = df[time_col].dt.time
    start = pd.to_datetime(SANE_WINDOW_START).time()
    end = pd.to_datetime(SANE_WINDOW_END).time()
    outside = ~times.between(start, end)
    if outside.any():
        pct = 100 * outside.sum() / len(df)
        report.warnings.append(
            f"{outside.sum()} rows ({pct:.2f}%) have `{time_col}` outside "
            f"{SANE_WINDOW_START}-{SANE_WINDOW_END} — if this is a large "
            "fraction, treat the whole file as suspect rather than just "
            "filtering rows (see Jan 2022 in research_log.md as precedent)."
        )


def validate_imbalance_file(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()

    missing_cols = set(IMBALANCE_EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        report.errors.append(f"Missing expected columns: {sorted(missing_cols)}")
        return report

    bad_side = ~df["Side"].isin([1, -1])
    if bad_side.any():
        report.errors.append(f"{bad_side.sum()} rows have Side outside {{1, -1}}.")

    bad_flip = ~df["FlipOn"].isin([0, 1])
    if bad_flip.any():
        report.errors.append(f"{bad_flip.sum()} rows have FlipOn outside {{0, 1}}.")

    if (df["PairOff"] < 0).any():
        report.errors.append("Negative PairOff values found.")

    if (df["LastSale"] <= 0).any():
        report.errors.append("Non-positive LastSale values found.")

    if df["Symbol"].isna().any():
        report.errors.append("Null Symbol values found.")

    dupes = df.duplicated(subset=["Symbol", "SaleTime"]).sum()
    if dupes:
        report.warnings.append(
            f"{dupes} duplicate (Symbol, SaleTime) rows — confirm whether "
            "this is expected (e.g. multiple updates within the same "
            "reported second)."
        )

    _check_time_window(df, "SaleTime", report)

    # flip_on consistency spot-check
    if {"Symbol", "SaleTime", "Side", "FlipOn"}.issubset(df.columns):
        df_sorted = df.sort_values(["Symbol", "SaleTime"])
        prev_side = df_sorted.groupby("Symbol")["Side"].shift(1)
        expected_flip = (df_sorted["Side"] != prev_side).astype("Int64")
        has_prior = prev_side.notna()
        mismatch = has_prior & (expected_flip != df_sorted["FlipOn"])
        if mismatch.any():
            report.warnings.append(
                f"{mismatch.sum()} rows where FlipOn doesn't match a simple "
                "side-change check vs. the prior tick for that Symbol — "
                "note this check doesn't reset at day boundaries, so some "
                "mismatches are expected there; investigate mismatches "
                "*within* a single day more closely."
            )

    return report


def validate_alerts_file(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()

    missing_cols = set(ALERTS_EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        report.errors.append(f"Missing expected columns: {sorted(missing_cols)}")
        return report

    bad_side = ~df["Side"].isin(["Buy", "Sell"])
    if bad_side.any():
        report.errors.append(f"{bad_side.sum()} rows have Side outside {{Buy, Sell}}.")

    if (df["Amount"] < 0).any():
        report.errors.append("Negative Amount values found.")

    if pd.api.types.is_datetime64_any_dtype(df["EntryDate"]) and pd.api.types.is_datetime64_any_dtype(
        df["EndDate"]
    ):
        bad_order = df["EndDate"] < df["EntryDate"]
        if bad_order.any():
            report.errors.append(f"{bad_order.sum()} rows have EndDate before EntryDate.")

    _check_time_window(df, "EntryDate", report)

    return report


def load_and_validate_imbalances(path: str) -> tuple[pd.DataFrame, ValidationReport]:
    df = pd.read_csv(path, parse_dates=["SaleTime"])
    return df, validate_imbalance_file(df)


def load_and_validate_alerts(path: str) -> tuple[pd.DataFrame, ValidationReport]:
    df = pd.read_csv(path, parse_dates=["EntryDate", "EndDate"])
    return df, validate_alerts_file(df)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3 or sys.argv[1] not in ("imbalances", "alerts"):
        print("Usage: python data_validation.py <imbalances|alerts> <path_to_csv>")
        sys.exit(1)

    loader = load_and_validate_imbalances if sys.argv[1] == "imbalances" else load_and_validate_alerts
    df, report = loader(sys.argv[2])
    print(report)
