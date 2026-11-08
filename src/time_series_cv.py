"""
Purged, embargoed walk-forward cross-validation for intraday financial data.

Why not sklearn's KFold or TimeSeriesSplit directly:
- Plain KFold shuffles rows, which for intraday data means training on
  minutes that are temporally *after* the minutes you're testing on — pure
  look-ahead leakage.
- Plain TimeSeriesSplit fixes ordering but doesn't account for the fact
  that a "label" here (return from t to close) uses information up to the
  close, so a training sample and a nearby test sample can still overlap
  in the information they depend on even when their timestamps don't
  overlap. This is the same leakage problem described in Lopez de Prado's
  "Advances in Financial Machine Learning" (purging / embargo).

This module implements:
- Contiguous walk-forward folds (train on the past, test on the future
  block immediately after it) — never randomly shuffled.
- Purging: drop training samples whose label window overlaps the test
  fold's time range.
- Embargo: drop a buffer of training samples immediately after each test
  fold, to prevent leakage from serial correlation bleeding across the
  train/test boundary in the other direction.

No real data is referenced or embedded in this file — it's generic and
gets mirrored into public_portfolio_repo/ unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class PurgedWalkForwardCV:
    """
    Generates (train_idx, test_idx) index arrays for walk-forward CV with
    purging and embargo.

    Parameters
    ----------
    n_splits : int
        Number of walk-forward folds.
    embargo_frac : float
        Fraction of each fold's sample count to embargo immediately after
        the test fold, expressed as a fraction of total samples (e.g. 0.01
        = embargo 1% of total rows after each test fold).
    label_horizon : pd.Timedelta or None
        How far forward a label looks (e.g. "return from t to close" has a
        horizon that varies intraday; for a fixed-horizon label pass the
        Timedelta here). Used for purging: any training sample whose
        [timestamp, timestamp + label_horizon] window overlaps the test
        fold's time range gets dropped from training. If your label
        horizon is variable (e.g. "to the close" rather than a fixed
        delta), pass a per-row Series of label end-times via
        `label_end_times` in `split()` instead and leave this None.
    """

    n_splits: int = 5
    embargo_frac: float = 0.01
    label_horizon: "pd.Timedelta | None" = None

    def split(
        self,
        timestamps: pd.Series,
        label_end_times: "pd.Series | None" = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Parameters
        ----------
        timestamps : pd.Series
            Timestamp for each row, sorted ascending, same index as your
            feature/label frame.
        label_end_times : pd.Series or None
            Per-row timestamp of when each row's label is fully determined
            (e.g. the close time for a "return to close" label). If None,
            falls back to `timestamp + label_horizon`.

        Yields
        ------
        (train_idx, test_idx) : positional integer index arrays, suitable
        for `.iloc[]`.
        """
        n = len(timestamps)
        if n < self.n_splits * 2:
            raise ValueError("Not enough samples for the requested number of splits.")

        ts = timestamps.reset_index(drop=True)
        if label_end_times is None:
            if self.label_horizon is None:
                raise ValueError(
                    "Must supply either label_end_times or label_horizon."
                )
            label_end = ts + self.label_horizon
        else:
            label_end = label_end_times.reset_index(drop=True)

        fold_bounds = np.linspace(0, n, self.n_splits + 1, dtype=int)
        embargo_n = max(1, int(round(n * self.embargo_frac)))

        for i in range(self.n_splits - 1):
            test_start, test_end = fold_bounds[i], fold_bounds[i + 1]
            test_idx = np.arange(test_start, test_end)

            test_time_start = ts.iloc[test_start]
            test_time_end = ts.iloc[test_end - 1]

            # Candidate training pool: everything outside the test fold.
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_start:test_end] = False

            # Purge: drop training rows whose label window overlaps the
            # test fold's time range in either direction.
            overlaps = (ts < test_time_end) & (label_end > test_time_start)
            train_mask &= ~overlaps.to_numpy()

            # Embargo: drop a buffer immediately after the test fold.
            embargo_end = min(n, test_end + embargo_n)
            train_mask[test_end:embargo_end] = False

            train_idx = np.nonzero(train_mask)[0]
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx

    def n_splits_effective(self) -> int:
        return self.n_splits - 1


def held_out_final_test_split(
    df: pd.DataFrame,
    timestamp_col: str,
    held_out_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split off a final untouched test set — the most recent `held_out_frac`
    of rows by time. This block should be looked at exactly once, after
    every question's methodology is finalized on the development set.

    Returns (development_df, held_out_df).
    """
    df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
    cutoff = int(round(len(df_sorted) * (1 - held_out_frac)))
    return df_sorted.iloc[:cutoff].copy(), df_sorted.iloc[cutoff:].copy()
