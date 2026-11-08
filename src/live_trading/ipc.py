"""File-based command/status protocol between the personal trading
dashboard (`trading_desk.py`, a Streamlit request/response process) and
the persistent background live-trading loop (`live_trader.py`, its own
long-running process). A Streamlit page only does work when someone has it
open and it happens to rerun -- that's not safe for managing open
positions, so the loop runs independently and the dashboard controls it
through two small JSON files instead of any in-process call.

`command.json` -- the dashboard (or `scripts/kill_switch.py`) writes a new
command; the loop polls this file every iteration and processes at most
one new command per poll, tracked by `id` so nothing is ever double-
processed. **The loop checks this file FIRST, before anything else, every
poll** -- a pending "kill" always wins over whatever else is in flight.

`status.json` -- the loop writes its current state every poll, for the
dashboard to display. The dashboard also independently queries
`AlpacaBroker` directly for ground truth on positions/equity -- this file
is a convenience/liveness signal, never the sole source of truth for
money numbers.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent.parent
IPC_DIR = REPO / "data" / "live_trading"
COMMAND_PATH = IPC_DIR / "command.json"
STATUS_PATH = IPC_DIR / "status.json"

VALID_ACTIONS = ("deploy", "pause", "resume", "kill")


def write_command(action: str, config_name: "str | None" = None, mode: str = "paper") -> dict:
    """`config_name`: a name previously saved via the research repo's
    `backtest_storage.save_config()` (same storage the Trading tab's
    Builder uses) -- looked up by `backtest_storage.load_config(name)`,
    not a raw file path."""
    if action not in VALID_ACTIONS:
        raise ValueError(f"unknown action {action!r} -- expected one of {VALID_ACTIONS}")
    IPC_DIR.mkdir(parents=True, exist_ok=True)
    command = {
        "id": str(uuid.uuid4()),
        "action": action,
        "config_name": config_name,
        "mode": mode,
        "issued_at": datetime.now(timezone.utc).isoformat(),
    }
    COMMAND_PATH.write_text(json.dumps(command, indent=2))
    return command


def read_pending_command(last_processed_id: "str | None") -> Optional[dict]:
    """Returns the command dict if `command.json` holds one not yet seen
    (by `id`), else None. Malformed/missing files are treated as "nothing
    pending" rather than raising -- a live trading loop must never crash
    on a transient read (e.g. the dashboard mid-write)."""
    if not COMMAND_PATH.exists():
        return None
    try:
        command = json.loads(COMMAND_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if command.get("id") == last_processed_id:
        return None
    return command


def write_status(state: str, active_config: "str | None", mode: "str | None",
                 positions: list, equity: float, realized_pnl: float, unrealized_pnl: float,
                 last_error: "str | None" = None) -> None:
    IPC_DIR.mkdir(parents=True, exist_ok=True)
    status = {
        "state": state, "active_config": active_config, "mode": mode,
        "positions": positions, "equity": equity,
        "realized_pnl": realized_pnl, "unrealized_pnl": unrealized_pnl,
        "last_poll": datetime.now(timezone.utc).isoformat(),
        "last_error": last_error,
    }
    STATUS_PATH.write_text(json.dumps(status, indent=2, default=str))


def read_status() -> Optional[dict]:
    if not STATUS_PATH.exists():
        return None
    try:
        return json.loads(STATUS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None
