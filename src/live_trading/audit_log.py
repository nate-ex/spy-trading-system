"""Append-only, structured audit log for every consequential live-trading
action (order intent, order result, kill-switch invocations). Committed to
git, not gitignored -- matching the research repo's own precedent that
logs of consequential actions are a tracked audit trail
(`logs/holdout_access_log.md`, `logs/research_log.md`), not a regenerable
cache. A record of every real order this system ever placed is the most
important audit trail it will produce.

One JSON object per line (JSONL), one file per UTC calendar day, so a
growing history never becomes one unbounded file and old days are trivial
to `git log`/diff independently.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
LOG_DIR = REPO / "logs"


def log_event(event_type: str, **fields) -> None:
    """Append one event. `event_type` examples: "order_intent",
    "order_result", "kill_switch_invoked", "position_closed". `fields` is
    freeform (whatever's relevant to that event type) -- always includes a
    UTC timestamp so events across a day are unambiguously orderable."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = LOG_DIR / f"live_orders_{day}.jsonl"
    record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event_type, **fields}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
