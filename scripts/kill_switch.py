"""Standalone kill switch -- flattens every open Alpaca position and
cancels every open order IMMEDIATELY, talking to Alpaca directly. Does
NOT depend on the background `run_live_trading.py` loop being alive or
responsive: even if that process has crashed or hung, this still works,
because it doesn't go through it at all.

Also writes a "kill" command to `data/live_trading/command.json` as a
courtesy, so the loop (if it IS still running and checks the file on its
next poll) also sees the halt signal and stops trying to place new orders
-- but the actual position-flattening above already happened regardless.

Usage:
    python scripts/kill_switch.py --paper
    python scripts/kill_switch.py --live      # requires explicit --live, never a default
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "data"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from live_trading import ipc  # noqa: E402
from live_trading.broker import AlpacaBroker  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--paper", action="store_true", help="Flatten the PAPER account.")
    mode.add_argument("--live", action="store_true", help="Flatten the REAL-MONEY account.")
    args = parser.parse_args()

    paper = args.paper
    print(f"[kill_switch] connecting to Alpaca ({'PAPER' if paper else 'LIVE / REAL MONEY'})...")
    broker = AlpacaBroker(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=paper)

    before = broker.positions()
    if not before:
        print("[kill_switch] no open positions found.")
    else:
        print(f"[kill_switch] closing {len(before)} open position(s): "
             f"{[(p['symbol'], p['qty']) for p in before]}")

    results = broker.close_all()
    print(f"[kill_switch] close_all() results: {results}")

    ipc.write_command("kill")
    print("[kill_switch] wrote kill command to data/live_trading/command.json "
         "(courtesy signal for the background loop, if it's running).")
    print("[kill_switch] done.")


if __name__ == "__main__":
    main()
