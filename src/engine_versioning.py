"""Cache-staleness helper for derived dashboard artifacts.

`dashboard_demo_grid.parquet` (`build_dashboard_demo.py`) and
`backtest_signals_dev.parquet` (`backtest_signals.py`) both bake one or
both frozen engines' (`stage1_density_engine.json`/
`touch_probability_engine.json`) predictions into a cached parquet file
on disk, gitignored, rebuilt only if missing (see
`bootstrap_dashboard_data.py`). If either engine is ever refit (e.g.
`refit_stage1_engine_pooled.py`, or a future addendum) on a machine that
already has these caches, they would silently keep serving predictions
from the OLD engine version -- neither REPLAY DEMO nor the Trading tab's
backtests would have any way to notice.

This module fingerprints both engine JSONs' current on-disk CONTENT
(not mtime -- content is what survives a fresh `git clone`/checkout
unchanged, mtime doesn't) and stamps it alongside each cache. `is_stale()`
is the check every cache-builder should call instead of a bare
`path.exists()`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STAGE1_ENGINE_PATH = REPO / "data" / "processed" / "stage1_density_engine.json"
TOUCH_ENGINE_PATH = REPO / "data" / "processed" / "touch_probability_engine.json"


def engine_fingerprint() -> str:
    """Short hash of both frozen engines' current on-disk content --
    changes iff either engine's parameters change (a refit/refreeze)."""
    h = hashlib.sha256()
    for path in (STAGE1_ENGINE_PATH, TOUCH_ENGINE_PATH):
        h.update(path.read_bytes() if path.exists() else b"<missing>")
    return h.hexdigest()[:16]


def stamp_path(artifact_path: Path) -> Path:
    return artifact_path.with_name(artifact_path.name + ".engine_fingerprint")


def write_stamp(artifact_path: Path) -> None:
    stamp_path(artifact_path).write_text(engine_fingerprint())


def is_stale(artifact_path: Path) -> bool:
    """True if `artifact_path` is missing, or its recorded fingerprint
    doesn't match the CURRENTLY loaded engines' fingerprint (built from an
    older/different engine fit). A cache with no stamp at all (e.g. one
    that predates this check) is conservatively treated as stale -- one
    extra rebuild is cheap; silently trusting an unstamped cache isn't."""
    if not artifact_path.exists():
        return True
    sp = stamp_path(artifact_path)
    if not sp.exists():
        return True
    return sp.read_text().strip() != engine_fingerprint()
