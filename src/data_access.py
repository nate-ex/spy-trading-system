"""
Single sanctioned interface for loading the development / held-out-test
datasets. Downstream research code should import from here rather than
opening `data/processed/*.parquet` directly, so that:

1. There is exactly one place that knows the file paths and split
   boundary, instead of every script re-deriving or re-filtering it.
2. Every access to the held-out set is logged to
   `logs/holdout_access_log.md` -- RESEARCH_CHARTER.md requires that
   dataset to be touched at most once, after every question's methodology
   is finalized on the development set. Logging makes that auditable
   rather than relying on discipline alone.

Run `src/split_dataset.py` first (or whenever `feature_table.parquet`
changes) to produce `development.parquet` / `holdout_test.parquet`.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DEVELOPMENT_PATH = PROCESSED_DIR / "development.parquet"
HOLDOUT_PATH = PROCESSED_DIR / "holdout_test.parquet"
HOLDOUT_ACCESS_LOG = REPO_ROOT / "logs" / "holdout_access_log.md"

_LOG_HEADER = (
    "# Held-Out Test Set Access Log\n\n"
    "Auto-appended by `src/data_access.py::load_holdout_test()` every time "
    "the held-out final test set is loaded. Per `RESEARCH_CHARTER.md`, this "
    "dataset should be touched at most once, after every question's "
    "methodology is finalized on the development set -- this log makes that "
    "auditable rather than just relying on convention.\n\n"
    "| Timestamp (UTC) | Calling script | Stated purpose |\n"
    "|---|---|---|\n"
)


def load_development() -> pd.DataFrame:
    """Load the development dataset. Unrestricted -- not logged."""
    if not DEVELOPMENT_PATH.exists():
        raise FileNotFoundError(
            f"{DEVELOPMENT_PATH} not found -- run src/split_dataset.py first."
        )
    return pd.read_parquet(DEVELOPMENT_PATH)


def load_holdout_test(purpose: str) -> pd.DataFrame:
    """
    Load the held-out final test dataset.

    `purpose` is required (not optional/defaulted) and is written to
    `logs/holdout_access_log.md` along with the calling script and a UTC
    timestamp. This dataset should be touched at most once per research
    question, after development-set methodology is fully finalized -- see
    RESEARCH_CHARTER.md and PRD.md's "Held-out final test split" note.
    """
    if not purpose or not purpose.strip():
        raise ValueError(
            "load_holdout_test() requires a non-empty `purpose` describing "
            "why the held-out set is being accessed right now."
        )
    if not HOLDOUT_PATH.exists():
        raise FileNotFoundError(
            f"{HOLDOUT_PATH} not found -- run src/split_dataset.py first."
        )

    caller_frame = inspect.stack()[1]
    caller = Path(caller_frame.filename).name

    df = pd.read_parquet(HOLDOUT_PATH)
    _log_holdout_access(caller=caller, purpose=purpose.strip())
    return df


def _log_holdout_access(caller: str, purpose: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    HOLDOUT_ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    is_new = not HOLDOUT_ACCESS_LOG.exists()
    with open(HOLDOUT_ACCESS_LOG, "a", encoding="utf-8") as f:
        if is_new:
            f.write(_LOG_HEADER)
        f.write(f"| {timestamp} | `{caller}` | {purpose} |\n")
