"""Save/reload strategy configs, sweep results, and the Compare tab's
running comparison table. Local files only (these are derived-from-real-data
research artifacts, private-repo-only per PRIVACY_AND_GIT_POLICY.md, same
tier as the rest of `data/processed/`).

Why the comparison table and last-sweep-results need their own
save/load functions (not just `st.session_state`, the way `bt_last_result`
is handled): LIVE mode's auto-refresh does a FULL BROWSER RELOAD every
poll (Streamlit has no native autorefresh -- see app.py's
`_inject_autorefresh`), which starts a brand-new session with an empty
`st.session_state` each time. A user parked on the Trading tab comparing
strategies while LIVE mode auto-refreshes in the background would
otherwise lose that comparison table every refresh cycle, not just
between days -- `dashboard_trading.py` persists both to these files on
every change and reloads them on init specifically to survive that."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from strategy_components import StrategyConfig

BACKTEST_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "backtests"
CONFIGS_DIR = BACKTEST_DIR / "configs"
RESULTS_DIR = BACKTEST_DIR / "results"
COMPARISON_PATH = BACKTEST_DIR / "comparison.json"
LAST_SWEEP_NAME = "_autosave_last_sweep"


def list_saved_configs() -> list[str]:
    if not CONFIGS_DIR.exists():
        return []
    return sorted(p.stem for p in CONFIGS_DIR.glob("*.json"))


def save_config(config: StrategyConfig) -> Path:
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in config.name)
    return config.save(CONFIGS_DIR / f"{safe_name}.json")


def load_config(name: str) -> StrategyConfig:
    return StrategyConfig.load(CONFIGS_DIR / f"{name}.json")


def delete_config(name: str) -> None:
    path = CONFIGS_DIR / f"{name}.json"
    if path.exists():
        path.unlink()


def save_sweep_results(results: pd.DataFrame, sweep_name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in sweep_name)
    path = RESULTS_DIR / f"{safe_name}.parquet"
    results.to_parquet(path, index=False)
    return path


def list_saved_sweeps() -> list[str]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(p.stem for p in RESULTS_DIR.glob("*.parquet"))


def load_sweep_results(sweep_name: str) -> pd.DataFrame:
    return pd.read_parquet(RESULTS_DIR / f"{sweep_name}.parquet")


def list_named_sweeps() -> list[str]:
    """Like `list_saved_sweeps()` but excludes the auto-saved "last sweep"
    slot (`LAST_SWEEP_NAME`) -- for a "load a PREVIOUSLY NAMED sweep"
    picker where the always-there autosave slot would just be noise."""
    return [n for n in list_saved_sweeps() if n != LAST_SWEEP_NAME]


def save_comparison(comparison: dict) -> Path:
    """Persist the Compare tab's {strategy_name: metrics_dict} table to
    disk. Called on every add/clear so a LIVE-mode auto-refresh (full
    browser reload -> fresh `st.session_state`) or a next-day relaunch
    never silently drops it -- see this module's docstring."""
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_PATH.write_text(json.dumps(comparison, indent=2, default=float))
    return COMPARISON_PATH


def load_comparison() -> dict:
    """{} if nothing has been saved yet (fresh machine, or cleared)."""
    if not COMPARISON_PATH.exists():
        return {}
    return json.loads(COMPARISON_PATH.read_text())


def save_last_sweep(results: pd.DataFrame) -> Path:
    """Auto-persist the most recently RUN sweep (distinct from the
    user-triggered "Save sweep results" button, which saves under a
    user-chosen name) so a LIVE-mode auto-refresh or next-day relaunch
    doesn't drop the table the user is currently looking at."""
    return save_sweep_results(results, LAST_SWEEP_NAME)


def load_last_sweep() -> "pd.DataFrame | None":
    if LAST_SWEEP_NAME not in list_saved_sweeps():
        return None
    return load_sweep_results(LAST_SWEEP_NAME)
