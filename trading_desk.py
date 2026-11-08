"""SPY Trading Desk — personal live-trading dashboard.

Fully self-contained: the live probability read (Dashboard tab), the
strategy research/backtest/sweep/compare environment (Trading tab), and
a Deploy tab (pick a saved strategy, send it live on Alpaca -- paper
first, real money later -- watch it, kill it instantly) all live in this
repo's own `src/`. (Split off from the sibling `trademas-imbalance-
research-private` repo on 2026-07-31 -- see git history there for prior
shared-code provenance; this repo no longer depends on it at all.)

This dashboard never runs order-placement logic itself. It only reads/
writes small files (`data/live_trading/command.json`/`status.json`) that
the persistent background loop (`run_live_trading.py`, its own process)
watches -- see that file's docstring for why. Run both:

    python run_live_trading.py                        # separate terminal
    streamlit run trading_desk.py --server.port 8502

Launch:  streamlit run trading_desk.py --server.port 8502
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SRC = REPO / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "data"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

from dashboard_view import render_dashboard, render_trading_summary_tab  # noqa: E402

from live_trading import ipc  # noqa: E402


def render_deploy_tab(st, regime_label, regime_emoji, regime_color, touch_preds,
                      agg_imbalance, imb_z) -> None:
    import backtest_storage

    st.subheader("Deploy")
    st.caption("Send a strategy you've already backtested (Trading tab) live on Alpaca — "
              "paper first, real money once you're ready.")

    status = ipc.read_status()
    _render_status_panel(st, status)

    st.divider()
    st.markdown("**Deploy a strategy**")
    saved = backtest_storage.list_saved_configs()
    if not saved:
        st.info("No saved strategies yet — build and save one in the Trading tab first.")
    else:
        col1, col2 = st.columns([2, 1])
        config_name = col1.selectbox("Strategy", saved)
        mode = col2.radio("Account", ["paper", "live"], horizontal=True)
        live_confirmed = True
        if mode == "live":
            live_confirmed = st.checkbox(
                "⚠️ I understand this places REAL orders with REAL money on the live Alpaca account.")
        deploy_disabled = mode == "live" and not live_confirmed
        if st.button("🚀 Deploy", disabled=deploy_disabled, type="primary"):
            ipc.write_command("deploy", config_name=config_name, mode=mode)
            st.success(f"Deploy command sent: {config_name!r} ({mode}). "
                      f"The background loop (run_live_trading.py) picks this up on its next poll.")

    st.divider()
    b1, b2, b3 = st.columns(3)
    if b1.button("⏸️  Pause (manage exits only)"):
        ipc.write_command("pause")
        st.info("Pause command sent.")
    if b2.button("▶️  Resume"):
        ipc.write_command("resume")
        st.info("Resume command sent.")

    st.divider()
    st.markdown("### 🔴 Kill switch")
    st.caption("Immediately flattens every open position and cancels every open order — "
              "talks to Alpaca directly, right here, not just via the background loop.")
    if st.button("🔴 KILL SWITCH — Flatten everything now", type="primary"):
        _execute_kill_switch(st, status)


def _render_status_panel(st, status: "dict | None") -> None:
    st.markdown("**Live status**")
    if status is None:
        st.caption("No status yet — the background loop (`run_live_trading.py`) hasn't polled yet, "
                  "or isn't running. Start it in a separate terminal.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("State", status.get("state", "?"))
    c2.metric("Mode", status.get("mode") or "—")
    c3.metric("Equity", f"${status.get('equity', 0):,.0f}")
    c4.metric("Unrealized P&L", f"${status.get('unrealized_pnl', 0):,.2f}")
    st.caption(f"Active strategy: {status.get('active_config') or '—'}  ·  "
              f"Last poll: {status.get('last_poll', '—')}")
    if status.get("last_error"):
        st.warning(f"Last error: {status['last_error']}")
    positions = status.get("positions") or []
    if positions:
        import pandas as pd
        st.dataframe(pd.DataFrame(positions), hide_index=True, width="stretch")
    else:
        st.caption("No open positions.")


def _execute_kill_switch(st, status: "dict | None") -> None:
    import os

    st.warning("Flattening everything now...")
    try:
        from live_trading.broker import AlpacaBroker

        paper = (status or {}).get("mode", "paper") != "live"
        broker = AlpacaBroker(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=paper)
        results = broker.close_all()
        ipc.write_command("kill")
        st.success(f"Kill switch executed. Closed: {results}")
    except Exception as exc:
        st.error(f"Kill switch failed to reach Alpaca directly: {exc}\n\n"
                "Run `python scripts/kill_switch.py --paper` (or `--live`) from a terminal "
                "as a fallback — it doesn't depend on this dashboard or the background loop.")


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="SPY Trading Desk", layout="wide",
                       initial_sidebar_state="expanded")
    render_dashboard(st, extra_tabs=[
        ("📊  Trading", render_trading_summary_tab),
        ("🚀  Deploy", render_deploy_tab),
    ], spy_source_mode="alpaca")  # never attempt Bloomberg here -- that's
    # a research-desk-machine thing (see dashboard_view.render_dashboard's
    # docstring). This dashboard always goes straight to Alpaca.


if __name__ == "__main__":
    main()
