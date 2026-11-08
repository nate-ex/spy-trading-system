"""
Physically separate the canonical feature table into development and
held-out-test parquet files, plus a manifest recording exactly how and
when the split was made.

Why a physical split, not just filtering `feature_table.parquet` by its
`split` column: RESEARCH_CHARTER.md requires the held-out final test set
to be touched at most once, after every question's methodology is
finalized on the development set. Relying on every future script to
remember `df[df["split"] == "development"]` makes accidental leakage one
missed filter away. `data_access.py` is the sanctioned way to load either
file going forward -- this script only needs to be re-run if
`feature_table.parquet` itself changes (e.g. a data-quality fix upstream).

`feature_table.parquet` remains the canonical, regenerable source of
truth (built by `build_master_dataset.py` as of Phase 3 -- see
RESEARCH_CHARTER.md's revised "Long-term target" -- which imports and
extends `build_feature_table.py`'s base layer with every other feature
family and multi-horizon targets); `development.parquet` and
`holdout_test.parquet` are derived, physically separate artifacts.

Run from anywhere (paths are resolved relative to this file):
    python split_dataset.py
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FEATURE_TABLE_PATH = PROCESSED_DIR / "feature_table.parquet"
DEVELOPMENT_PATH = PROCESSED_DIR / "development.parquet"
HOLDOUT_PATH = PROCESSED_DIR / "holdout_test.parquet"
MANIFEST_PATH = PROCESSED_DIR / "split_manifest.json"


def _git_commit_hash() -> str | None:
    """Best-effort HEAD hash -- None (not an error) if git isn't available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def split_dataset(feature_table_path: Path = FEATURE_TABLE_PATH) -> dict:
    df = pd.read_parquet(feature_table_path)
    if "split" not in df.columns:
        raise ValueError(
            f"{feature_table_path} has no 'split' column -- run "
            "build_feature_table.add_final_test_split() first."
        )

    dev = df[df["split"] == "development"].sort_values("time").reset_index(drop=True)
    held = df[df["split"] == "held_out"].sort_values("time").reset_index(drop=True)

    if len(dev) == 0 or len(held) == 0:
        raise ValueError("Split produced an empty development or held-out set.")

    dev.to_parquet(DEVELOPMENT_PATH, index=False)
    held.to_parquet(HOLDOUT_PATH, index=False)

    manifest = {
        "split_boundary_date": str(held["date"].min().date()),
        "development": {
            "path": str(DEVELOPMENT_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
            "date_range": [str(dev["date"].min().date()), str(dev["date"].max().date())],
            "n_rows": int(len(dev)),
            "n_days": int(dev["date"].nunique()),
        },
        "holdout_test": {
            "path": str(HOLDOUT_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
            "date_range": [str(held["date"].min().date()), str(held["date"].max().date())],
            "n_rows": int(len(held)),
            "n_days": int(held["date"].nunique()),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit_hash": _git_commit_hash(),
        "source_file": str(feature_table_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "methodology": (
            "Day-aware split: held_out_final_test_split() (time_series_cv.py) "
            "takes the most recent 15% of rows by time, then "
            "build_feature_table.add_final_test_split() snaps the cutoff "
            "forward to the start of a full trading day so no single day's "
            "cross-section is split across development/held-out. See "
            "PRD.md 'Held-out final test split' and "
            "logs/data_quality_report.md section 7."
        ),
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    manifest = split_dataset()
    print(json.dumps(manifest, indent=2))
