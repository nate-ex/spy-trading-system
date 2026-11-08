"""Entrypoint for the persistent background live-trading loop.

Run this once (e.g. in its own terminal, or as a background process) and
leave it running. It starts idle and does nothing until the dashboard (or
`scripts/kill_switch.py`) writes a "deploy" command to
`data/live_trading/command.json`. Stop it with Ctrl-C, or the kill switch
(which works even if this process has become unresponsive, by talking to
Alpaca directly instead of relying on this loop noticing the command).

Usage:
    python run_live_trading.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "data"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent / ".env")

from live_trading.live_trader import LiveTrader  # noqa: E402


def main() -> None:
    trader = LiveTrader()
    print("[run_live_trading] idle -- waiting for a deploy command "
          "(data/live_trading/command.json). Ctrl-C to stop.")
    try:
        trader.run_forever()
    except KeyboardInterrupt:
        print("\n[run_live_trading] interrupted -- flattening any open positions before exit.")
        if trader.manager is not None:
            try:
                row = trader._poll_live_state()
                trader.manager.force_flatten(row)
            except Exception as exc:
                print(f"[run_live_trading] flatten-on-exit failed ({exc}); "
                      f"falling back to a direct close_all().")
                trader.manager.broker.close_all()


if __name__ == "__main__":
    main()
