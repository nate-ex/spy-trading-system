"""SPY Probability Desk -- shared dashboard rendering, extracted out of
`app.py` (unchanged behavior, just relocated) so a second, separate
Streamlit app (a personal trading desk, in its own private repo, never
pulled onto a machine that only needs the research dashboard) can reuse
the exact same live/replay probability-dashboard experience without
duplicating it.

`render_dashboard(st, extra_tabs=...)` renders the header, mode toggle
(REPLAY DEMO / LIVE STREAMING), state loading, and the "Dashboard" tab
(probability cards, market context, Advanced panel) -- then renders one
more tab per `(label, render_fn)` pair in `extra_tabs`, each called as
`render_fn(st, regime_label, regime_emoji, regime_color, touch_preds,
agg_imbalance, imb_z)`. `app.py` passes a single extra tab
(`render_trading_summary_tab`, the existing Trading tab); a second
dashboard can pass that same one plus its own additional tabs (e.g. a
Deploy/kill-switch tab) with no duplication of any of the code below.
"""

from __future__ import annotations

from datetime import datetime, time
from pathlib import Path
from typing import Callable

import pandas as pd

from live_signal_state import (
    append_live_imbalance_snapshot,
    live_imbalance_zscore_intraday,
    live_imbalance_zscore_tod,
    live_predictions,
    load_live_imbalance_history,
    touch_predictions,
)

REPO = Path(__file__).resolve().parent.parent
GRID = REPO / "data" / "processed" / "dashboard_demo_grid.parquet"
MOVERS = REPO / "data" / "processed" / "dashboard_demo_movers.parquet"
ENGINE_JSON = REPO / "data" / "processed" / "stage1_density_engine.json"
GATE_HIGH = 0.65      # P_touch(15m) gate for HIGH-DISPERSION opportunity
PLOT_HORIZONS = ["5min", "10min", "15min", "20min", "30min"]
TAU_BPS = 10.0
HORIZON_LABEL = {"5min": "Next 5 Minutes", "10min": "Next 10 Minutes", "15min": "Next 15 Minutes",
                 "20min": "Next 20 Minutes", "30min": "Next 30 Minutes"}
REGIME_COLOR = {"🟢": "#12703a", "🟡": "#8a6d1a", "⚪": "#444", "🔴": "#a83232"}


# ----------------------------- pure helpers (testable) -----------------------------
def load_grid() -> pd.DataFrame:
    import json  # noqa: F401  (kept import-local to match original module's lazy style)
    g = pd.read_parquet(GRID)
    g["time"] = pd.to_datetime(g["time"]); g["date"] = pd.to_datetime(g["date"])
    return g


def load_movers() -> pd.DataFrame:
    m = pd.read_parquet(MOVERS)
    m["time"] = pd.to_datetime(m["time"]); m["date"] = pd.to_datetime(m["date"])
    return m


def load_engine_params() -> dict:
    import json
    return json.loads(Path(ENGINE_JSON).read_text())


def opportunity_label(p_touch_15m: float, gate: float = GATE_HIGH) -> tuple[str, str]:
    """Return (label, emoji) for the regime banner, gated on the 15-minute
    TOUCH probability (the number a trader actually watching the screen
    would experience), not the terminal-return P_tail."""
    if p_touch_15m >= gate:
        return "HIGH DISPERSION OPPORTUNITY", "🟢"
    if p_touch_15m >= gate - 0.15:
        return "ELEVATED", "🟡"
    return "QUIET MARKET", "⚪"


def live_regime(touch_preds: dict, spy_status: "str | None", gate: float = GATE_HIGH) -> tuple[str, str]:
    """(label, emoji) for the regime banner -- `opportunity_label` when
    real touch predictions exist; otherwise a state describing WHY not,
    distinguishing the expected/transient "waiting for today's first SPY
    bar" case from a genuine feed error, so the banner never just goes
    blank or shows a stale/misleading number when SPY data isn't
    available yet. `spy_status` is None on the REPLAY path and whenever
    LIVE mode has a real reading (see `render_dashboard`)."""
    if touch_preds:
        ptail_gate_h = "15min" if "15min" in touch_preds else list(touch_preds.keys())[-1]
        return opportunity_label(touch_preds[ptail_gate_h]["P_touch"], gate)
    if spy_status == "waiting":
        return "WAITING FOR SPY DATA", "🟡"
    if spy_status == "error":
        return "LIVE SPY FEED ERROR", "🔴"
    # At/after 16:00 no horizon has an exit on/before the MOC cross
    # (project hard constraint — see stage1_density_engine.py's
    # docstring), so the engine has nothing to predict yet/anymore.
    return "MARKET AT CLOSE", "⚪"


def minutes_between(now_t: time, target_t: time) -> float:
    base = datetime(2000, 1, 1)
    return (datetime.combine(base, target_t) - datetime.combine(base, now_t)).total_seconds() / 60.0


def fmt_usd(x: float) -> str:
    a = abs(x); sign = "+" if x >= 0 else "-"
    if a >= 1e9:
        return f"{sign}${a/1e9:.2f}B"
    if a >= 1e6:
        return f"{sign}${a/1e6:.0f}M"
    return f"{sign}${a:,.0f}"


def price_imbalance_figure(price_df: pd.DataFrame, imbalance_df: pd.DataFrame):
    """SPY price trajectory (session so far) overlaid with net NYSE
    imbalance -- same shape in both REPLAY and LIVE mode: `price_df` needs
    `time`/`close`; `imbalance_df` needs `time`/`aggregate_dollar_imbalance`."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_df["time"], y=price_df["close"], name="SPY", yaxis="y1",
                             line=dict(color="#1f77b4", width=2)))
    if not imbalance_df.empty:
        fig.add_trace(go.Scatter(x=imbalance_df["time"], y=imbalance_df["aggregate_dollar_imbalance"] / 1e9,
                                 name="Net S&P 500 Imbal. ($B)", yaxis="y2",
                                 line=dict(color="#d62728", width=1.5, dash="dot")))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(title="SPY"), yaxis2=dict(title="Imbal. $B", overlaying="y", side="right"),
                      legend=dict(orientation="h", y=1.1), template="plotly_white",
                      # uirevision: a CONSTANT value (not tied to the data) tells
                      # Plotly.js to preserve zoom/pan/legend-toggle state across
                      # figure updates instead of resetting to autorange -- without
                      # this, every LIVE auto-refresh (every 2s) snaps any zoom you
                      # did back out, making it impossible to actually focus on part
                      # of the chart while it keeps live-updating.
                      uirevision="spy_price_imbalance_chart")
    return fig


def replay_predictions(row: pd.Series, engine_params: dict, horizons: list = PLOT_HORIZONS) -> dict:
    """Per-horizon {sigma_bps, eta, lambda, mu_bps, P_tail, P_upside,
    P_downside} from a precomputed demo-grid row -- TERMINAL-return
    engine, shown in the Advanced panel. sigma/P_tail/P_upside/P_downside
    are the row's own (time-varying) values; eta/lambda/mu come straight
    from the frozen engine params (per-horizon CONSTANTS)."""
    out = {}
    for h in horizons:
        p = engine_params["by_horizon"][h]
        eta = row.get(f"eta_{h}", p["skew_eta"]); lam = row.get(f"lambda_{h}", p["skew_lambda"])
        out[h] = {"sigma_bps": float(row[f"sigma_{h}_bps"]), "eta": float(eta), "lambda": float(lam),
                  "mu_bps": float(p["loc_mu"]) * 1e4,
                  "P_tail": float(row[f"ptail_{h}"]),
                  "P_upside": float(row.get(f"p_upside_{h}", row[f"ptail_{h}"] / 2)),
                  "P_downside": float(row.get(f"p_downside_{h}", row[f"ptail_{h}"] / 2))}
    return out


# ----------------------------- presentation helpers -----------------------------
def _inject_css(st) -> None:
    st.markdown("""
    <style>
    div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
    div[data-testid="stMetricLabel"] { font-size: 0.85rem; opacity: 0.75; }
    .prob-card {
        border-radius: 14px; padding: 22px 18px; text-align: center;
        background: #f5f7fa;
        border: 1px solid rgba(128,128,128,0.25);
        color: #1a1a2e;
    }
    /* Fixed light card + dark text, independent of the Streamlit theme
       (light/dark) actually active -- `var(--background-color, ...)`
       previously used here isn't a CSS variable Streamlit actually sets,
       so it silently fell back to a near-white background while the
       label/subtitle text (no explicit color) inherited the page's
       theme text color -- white-on-near-white in dark theme, invisible
       except for the percentage (which already had its own inline
       color). Making the card colors fixed/explicit, not theme-derived,
       guarantees contrast regardless of theme. */
    .prob-card .prob-h { font-size: 1.0rem; font-weight: 600; opacity: 0.75; letter-spacing: 0.02em; color: inherit; }
    .prob-card .prob-pct { font-size: 3.4rem; font-weight: 800; line-height: 1.1; margin: 6px 0; }
    .prob-card .prob-sub { font-size: 0.92rem; opacity: 0.7; color: inherit; }
    .regime-banner {
        padding: 16px 22px; border-radius: 10px; color: white;
        font-size: 1.5rem; font-weight: 700; margin-bottom: 14px;
    }
    </style>
    """, unsafe_allow_html=True)


def _prob_card_html(horizon_label: str, pct: float, tau_bps: float, color: str) -> str:
    return (
        f"<div class='prob-card'>"
        f"<div class='prob-h'>{horizon_label}</div>"
        f"<div class='prob-pct' style='color:{color}'>{pct:.0%}</div>"
        f"<div class='prob-sub'>Chance SPY moves at least &plusmn;{tau_bps:.0f} basis points</div>"
        f"</div>"
    )


def _prob_color(pct: float) -> str:
    if pct >= GATE_HIGH:
        return "#12703a"
    if pct >= GATE_HIGH - 0.15:
        return "#8a6d1a"
    return "#1f77b4"


def _movers_display(mv) -> pd.DataFrame:
    if isinstance(mv, pd.DataFrame) and not mv.empty:
        return mv[["ticker", "imb_usd"]].assign(imb_usd=mv["imb_usd"].map(fmt_usd))
    if isinstance(mv, list) and mv:
        return pd.DataFrame(mv, columns=["ticker", "imb_usd"]).assign(imb_usd=lambda d: d["imb_usd"].map(fmt_usd))
    return pd.DataFrame(columns=["ticker", "imb_usd"])


def _render_advanced_panel(st, preds: dict, touch_preds: dict, features: dict) -> None:
    import plotly.graph_objects as go
    import dashboard_density as dd

    st.markdown("**Touch-probability model (Weibull-vs-lognormal comparison led to a log-normal "
              "shape — see `MODEL_EQUATION.md`)**")
    tcols = st.columns(len(PLOT_HORIZONS))
    for col, h_name in zip(tcols, PLOT_HORIZONS):
        if h_name not in touch_preds:
            continue
        t = touch_preds[h_name]
        with col:
            st.markdown(f"**{h_name}**")
            st.metric("P_touch", f"{t['P_touch']:.1%}")
            st.metric("Excursion scale", f"{t['scale_bps']:.1f} bps")
            st.caption(f"log-normal(s={t['lognorm_s']:.3f}, scale={t['lognorm_scale']:.3f})")

    st.divider()
    st.markdown("**Terminal-return distribution (Hansen skewed Student-t) — the ORIGINAL engine, "
              "unchanged. This is the return measured exactly at t+h, not a touch probability.**")
    fig = go.Figure()
    palette = {"5min": "#1f77b4", "10min": "#ff7f0e", "15min": "#2ca02c",
              "20min": "#d62728", "30min": "#9467bd"}
    for h_name in PLOT_HORIZONS:
        if h_name not in preds:
            continue
        p = preds[h_name]
        curve = dd.density_curve(p["mu_bps"], p["sigma_bps"], p["eta"], p["lambda"], tau_bps=TAU_BPS)
        color_h = palette.get(h_name, "#888")
        fig.add_trace(go.Scatter(x=curve["x_bps"], y=curve["pdf"], name=f"{h_name} density",
                                 line=dict(color=color_h, width=2)))
        for mask, opacity in [(curve["upside_mask"], 0.25), (curve["downside_mask"], 0.25)]:
            fig.add_trace(go.Scatter(x=curve["x_bps"][mask], y=curve["pdf"][mask], fill="tozeroy",
                                     mode="none", fillcolor=color_h, opacity=opacity, showlegend=False,
                                     hoverinfo="skip"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(title="forward return (bps)"), yaxis=dict(title="density"),
                      legend=dict(orientation="h", y=1.08), template="plotly_white")
    st.plotly_chart(fig, width="stretch")

    dcols = st.columns(len(PLOT_HORIZONS))
    for col, h_name in zip(dcols, PLOT_HORIZONS):
        if h_name not in preds:
            continue
        p = preds[h_name]
        with col:
            st.markdown(f"**{h_name}**")
            st.metric("P_tail (terminal)", f"{p['P_tail']:.0%}")
            st.metric("P_upside / P_downside", f"{p['P_upside']:.0%} / {p['P_downside']:.0%}")
            st.metric("σ (sigma)", f"{p['sigma_bps']:.1f} bps")
            st.metric("η / λ (eta / lambda)", f"{p['eta']:.1f} / {p['lambda']:+.3f}")

    st.divider()
    st.markdown("**Regression inputs (state features driving both engines' scale)**")
    st.json({k: features.get(k) for k in
            ["minutes_to_close", "realized_vol_cum", "realized_vol_30min", "vix_close_lagged"]})
    st.caption("⚠️ Terminal-return P_tail was found slightly conservative OOS in calm 2025 regimes "
              "(see HOLDOUT_AUDIT_REPORT.md) — shown as a gauge, not a to-the-decimal probability.")


def render_trading_summary_tab(st, regime_label, regime_emoji, regime_color, touch_preds,
                               agg_imbalance, imb_z) -> None:
    """The research dashboard's "Trading" tab: a live regime/opportunity
    snapshot + research notes, then the full strategy research &
    backtesting environment (`dashboard_trading.render_trading_tab`).
    Reusable as one of `render_dashboard`'s `extra_tabs` by any dashboard
    that wants this exact content (a second, personal dashboard can pass
    this AND its own additional tabs)."""
    st.subheader("Trading")
    st.caption("Reserved for future execution research — no live trading logic implemented.")

    st.markdown(
        f"<div class='regime-banner' style='background:{regime_color}'>"
        f"{regime_emoji}  {regime_label}</div>", unsafe_allow_html=True)

    st.markdown("**Opportunity probabilities**")
    ocols = st.columns(len(PLOT_HORIZONS))
    for col, h_name in zip(ocols, PLOT_HORIZONS):
        if h_name not in touch_preds:
            continue
        col.metric(HORIZON_LABEL[h_name], f"{touch_preds[h_name]['P_touch']:.0%}")

    st.markdown("**Current imbalance**")
    icols = st.columns(2)
    icols[0].metric("Net S&P 500 Imbalance", fmt_usd(agg_imbalance))
    icols[1].metric("Imbalance Z-Score", f"{imb_z:+.2f}" if pd.notna(imb_z) else "n/a")

    st.divider()
    st.markdown("**Research notes**")
    st.markdown(
        "- Direction (both conditional mean AND conditional skew) is a **martingale** in regular "
        "trading hours — reconfirmed across Q007/Q008/Q009. No forecastable up/down edge exists in "
        "this feature set.\n"
        "- Q008's gated structural rules (dispersion mean-reversion, cross-market momentum) were both "
        "**null** — the system correctly sizes to f\\*=0 (abstain) rather than force a directional bet.\n"
        "- Dispersion (what this dashboard's probability cards measure) **cannot be monetized with a "
        "linear SPY instrument and no direction** — it needs either a convex (options) instrument or "
        "a genuine direction source, both requiring new data the project doesn't yet have "
        "(options/dealer-gamma positioning is the highest-leverage next acquisition — see PRD.md "
        "Open Items)."
    )
    st.info("📍 Live snapshot above; the strategy research environment below is where rule-based "
           "ideas actually get built and backtested. Nothing here trades live.")

    st.divider()
    import dashboard_trading as dt
    dt.render_trading_tab(st)


def _replay_state(engine_params: dict, st):
    grid = load_grid()
    st.sidebar.title("🎛️  Desk Controls")
    days = sorted(grid["date"].dt.date.unique())
    sel_day = st.sidebar.selectbox("Replay session", days, index=len(days) - 1,
                                   format_func=lambda d: d.isoformat())
    day_grid = grid[grid["date"].dt.date == sel_day].sort_values("time").reset_index(drop=True)
    times = list(day_grid["time"].dt.strftime("%H:%M"))
    sel_time_str = st.sidebar.select_slider("Replay time (14:00→16:00)", options=times,
                                            value=times[min(len(times) - 1, len(times) // 2)])
    row = day_grid[day_grid["time"].dt.strftime("%H:%M") == sel_time_str].iloc[0]
    now = row["time"].to_pydatetime()
    preds = replay_predictions(row, engine_params)
    features = {"minutes_to_close": row["minutes_to_close"], "vix_close_lagged": row["vix_close_lagged"],
               "realized_vol_cum": row["realized_vol_cum"], "realized_vol_30min": row["realized_vol_30min"]}
    return now, preds, features, day_grid, sel_day, sel_time_str


def _load_live_spy_state(engine_params: dict, source: str = "auto"):
    """SPY price/probability state ONLY. `source` is passed straight
    through to `live_stage1_feed.fetch_today_spy_bars`: `"auto"` (Bloomberg
    if available, else Alpaca's free IEX feed -- the research dashboard's
    default, since a Bloomberg Terminal may or may not be on that machine)
    or `"alpaca"` (never even attempts Bloomberg -- the personal trading
    dashboard's setting, since a Bloomberg Terminal session is specifically
    a research-desk-machine thing, not something the trading dashboard
    should ever try or depend on). May raise
    `live_stage1_feed.NoLiveBarsYetError` (expected/transient -- no bars
    posted yet today, e.g. before/at market open) or any other Exception
    (a real connectivity/config problem) -- `render_dashboard`
    distinguishes the two to show a calm "waiting for data" message vs. a
    real error, and never lets either stop the rest of the page from
    rendering. The NYSE imbalance feed (TradeMas) is fetched entirely
    separately (`_fetch_live_imbalance_ticks`) since it has nothing to do
    with whether this succeeds."""
    import live_stage1_feed as lsf
    from stage1_density_engine import Stage1DensityEngine

    engine = Stage1DensityEngine.load()
    result = lsf.fetch_and_predict(engine, tau_bps=TAU_BPS, source=source)
    features = result["features"]
    now = features["time"] if isinstance(features["time"], datetime) else datetime.combine(
        datetime.today(), features["time"])
    preds = live_predictions(result["prediction"], engine_params) if result["warmed_up"] else {}
    return now, preds, features, result["warmed_up"], result["bars"], result["source_used"]


def _fetch_live_imbalance_ticks() -> tuple[list, "str | None"]:
    """Best-effort fetch of live NYSE imbalance ticks (TradeMas). Returns
    (ticks, status): status is None when ticks were returned, else a short
    human-readable reason to show instead ("fetch failed: ..." on a real
    error, "no data returned" on a clean empty response -- e.g. outside
    the feed's ~13:30-16:00 active window, see TradeMasImbalanceFeed's
    docstring). Never raises: whatever goes wrong here, the rest of LIVE
    mode (SPY price/probability panels) must still render."""
    from live_feeds import TradeMasImbalanceFeed

    try:
        ticks = TradeMasImbalanceFeed().fetch_once()
    except Exception as exc:
        return [], f"fetch failed: {exc}"
    if not ticks:
        return [], "no data returned"
    return ticks, None


def _live_aggregate_imbalance(ticks) -> float:
    from live_feeds import aggregate_dollar_imbalance

    return aggregate_dollar_imbalance(ticks) if ticks else 0.0


def _live_top_movers(ticks):
    from live_feeds import top_movers

    tm = top_movers(ticks) if ticks else {"buy": [], "sell": []}
    return tm["buy"], tm["sell"]


# ----------------------------- shared UI pieces -----------------------------
# Split into a fast "numbers" section (session caption through the Market
# Context metrics -- everything that's a live-updating number/probability/
# banner) and a slower "heavy" section (top movers table, the SPY-vs-
# imbalance chart, the Advanced panel, footer captions) so LIVE mode can
# refresh them on two different cadences: numbers every 1-2s, heavy visuals
# every several seconds. REPLAY mode (no timer at all) just calls both in
# sequence, identical to the old single-pass render.
def _render_numbers_ui(st, session_label: str, warmed_up: bool, is_live: bool,
                       spy_status: "str | None", imbalance_status: "str | None",
                       features: dict, cur_vol_bps: float, agg_imbalance: float,
                       imb_z: float, imb_z_tod: float, regime_label: str, regime_emoji: str,
                       regime_color: str, touch_preds: dict) -> None:
    st.caption(session_label)

    if warmed_up:
        # ---- 7 live metric cards ----
        m = st.columns(7)
        m[0].metric("Minutes to Close", f"{features.get('minutes_to_close', float('nan')):.0f}")
        m[1].metric("Current Volatility", f"{cur_vol_bps:.1f} bps")
        m[2].metric("30-Min Realized Vol", f"{features.get('realized_vol_30min', float('nan')) * 1e4:.1f} bps")
        m[3].metric("Cumulative Realized Vol", f"{features.get('realized_vol_cum', float('nan')) * 1e4:.1f} bps")
        m[4].metric("Net S&P 500 Imbalance", fmt_usd(agg_imbalance))
        m[5].metric("Imbalance Z-Score", f"{imb_z:+.2f}" if pd.notna(imb_z) else "n/a",
                   help="Same-day expanding std -- widens over the session, see Z-Score (ToD).")
        m[6].metric("Imbalance Z-Score (ToD)", f"{imb_z_tod:+.2f}" if pd.notna(imb_z_tod) else "n/a",
                   help="Cross-day, same-clock-time-matched std instead -- doesn't inflate "
                       "within a session. See strategy_components.py's zscore_source.")
    elif is_live:
        # SPY price/probability data isn't available right now (either
        # "waiting" -- expected/transient, self-heals on the next poll --
        # or "error" -- see the red box above). Either way, Market Context
        # (NYSE imbalance) below is a COMPLETELY SEPARATE feed and still
        # renders normally regardless of this.
        waiting_msg = ("⏳ No SPY bars yet today — expected before/at market "
                      "open, or if today's intraday bars haven't posted yet. This will "
                      "populate automatically once bars arrive.") if spy_status == "waiting" else (
                      "⚠️ SPY price/probability data is unavailable right now — see the error "
                      "above.")
        st.info(f"{waiting_msg} Market Context below (NYSE imbalance) is an independent feed "
               "and is unaffected by this.")

    st.markdown(
        f"<div class='regime-banner' style='background:{regime_color}'>"
        f"{regime_emoji}  {regime_label}</div>", unsafe_allow_html=True)

    if warmed_up:
        # ---- PRIMARY: touch-probability cards ----
        st.subheader("What is SPY likely to do next?")
        p_cols = st.columns(len(PLOT_HORIZONS))
        for col, h_name in zip(p_cols, PLOT_HORIZONS):
            if h_name not in touch_preds:
                col.info(f"{HORIZON_LABEL[h_name]}: past today's close.")
                continue
            pct = touch_preds[h_name]["P_touch"]
            col.markdown(_prob_card_html(HORIZON_LABEL[h_name], pct, TAU_BPS, _prob_color(pct)),
                        unsafe_allow_html=True)
        st.caption("Probability price touches the threshold at ANY point in the window — not "
                  "just where it ends up. Direction is a coin flip in RTH by finding "
                  "(Q008/Q009), so this is a magnitude/opportunity read, not a buy/sell signal.")

    st.divider()

    # ---- Market context: ALWAYS rendered in live mode, independent of
    # whether SPY data (above) is available -- NYSE imbalance (TradeMas)
    # and SPY price/probability (Bloomberg/Alpaca) are two unrelated
    # feeds. Top buy/sell names + the chart live in `_render_heavy_ui`
    # instead -- purely a code-organization split, both render on the
    # same fragment/timer as everything else here. ----
    st.subheader("Market Context")
    if is_live and imbalance_status:
        st.caption(f"ℹ️ No current imbalance data available ({imbalance_status}).")
    ic1, ic2 = st.columns(2)
    ic1.metric("💵 Net S&P 500 Imbalance", fmt_usd(agg_imbalance))
    ic2.metric("📈 Imbalance Z-Score", f"{imb_z:+.2f}" if pd.notna(imb_z) else "n/a")


def _render_heavy_ui(st, is_live: bool, warmed_up: bool, movers_buy, movers_sell,
                     price_df: pd.DataFrame, imbalance_hist_df: pd.DataFrame,
                     preds: dict, touch_preds: dict, features: dict) -> None:
    st.markdown("**Top buy / sell names**")
    c1, c2 = st.columns(2)
    c1.markdown("🟢 Buy")
    c1.dataframe(_movers_display(movers_buy), hide_index=True, width="stretch")
    c2.markdown("🔴 Sell")
    c2.dataframe(_movers_display(movers_sell), hide_index=True, width="stretch")

    st.markdown("**SPY vs. Net S&P 500 Imbalance (session so far)**")
    if not price_df.empty:
        st.plotly_chart(price_imbalance_figure(price_df, imbalance_hist_df), width="stretch",
                       key="spy_price_imbalance_chart")
    else:
        st.caption("No SPY price history yet.")
    if is_live and imbalance_hist_df.empty:
        st.caption("Imbalance trajectory starts accumulating from this point forward each poll.")

    if warmed_up:
        # ---- Advanced panel ----
        with st.expander("⚙️ Advanced — full statistical detail", expanded=False):
            _render_advanced_panel(st, preds, touch_preds, features)

    st.divider()
    st.caption("Touch-probability engine: development-calibrated, OOS PIT KS 0.019–0.024 "
              "(purged walk-forward CV). Terminal-return engine (Advanced panel): one-shot "
              "holdout-audited. Probabilistic market-state gauges, not a trading platform.")


# ----------------------------- main entry point -----------------------------
def render_dashboard(st, extra_tabs: "list[tuple[str, Callable]] | None" = None,
                     spy_source_mode: str = "auto") -> None:
    """Renders the full probability-dashboard experience: header, mode
    toggle, state loading, and the "Dashboard" tab -- plus one more tab per
    `(label, render_fn)` in `extra_tabs`, each called as `render_fn(st,
    regime_label, regime_emoji, regime_color, touch_preds, agg_imbalance,
    imb_z)`. Callers must have already set `st.set_page_config(...)`
    (a Streamlit constraint: it must be the first `st` call on the page,
    so this function can't do it itself without dictating page config to
    every caller).

    `spy_source_mode`: forwarded to `_load_live_spy_state`/
    `live_stage1_feed.fetch_today_spy_bars`. `"auto"` (default, used by
    the research dashboard) tries Bloomberg first, falls back to Alpaca.
    `"alpaca"` (used by the personal trading dashboard) never attempts
    Bloomberg at all -- a Bloomberg Terminal session is specific to the
    research desk's machine, not something the trading dashboard should
    ever try or depend on. All user-facing copy below (sidebar caption,
    waiting/error messages) is written to match whichever mode is active,
    never mentioning Bloomberg when it isn't in play."""
    import pytz
    from touch_probability_engine import TouchProbabilityEngine

    _inject_css(st)

    # ---- header ----
    # Read/write the mode via st.query_params so an explicit REPLAY DEMO
    # selection survives a real browser refresh (a genuine full-page
    # reload, not the LIVE fragment's own in-place polling).
    modes = ["🟡 REPLAY DEMO", "🔴 LIVE STREAMING"]
    # Defaults to LIVE STREAMING -- this dashboard is watched live while
    # trading, not opened to browse REPLAY DEMO first. Also the only
    # option that works at all on a live-only machine with no REPLAY DEMO
    # artifacts (e.g. a trading-floor box provisioned without this
    # project's historical research data -- see bootstrap_dashboard_data.
    # py's live-only path). An explicit `?mode=demo` (from a prior REPLAY
    # DEMO selection surviving a refresh) is the only way to land on
    # REPLAY DEMO by default.
    default_idx = 0 if st.query_params.get("mode") == "demo" else 1
    title_col, toggle_col = st.columns([3, 1.4])
    title_col.title("📈 SPY Probability Desk")
    stream_mode = toggle_col.radio(
        "Feed mode", modes, index=default_idx, horizontal=False, label_visibility="collapsed")
    is_live = stream_mode.startswith("🔴")
    st.query_params["mode"] = "live" if is_live else "demo"

    engine_params = load_engine_params()
    touch_engine = TouchProbabilityEngine.load()

    if spy_source_mode == "alpaca":
        st.sidebar.caption("🟡 REPLAY DEMO needs no setup. 🔴 LIVE needs `.env` TradeMas credentials "
                          "for the imbalance panel and `.env` Alpaca credentials (free, no Terminal "
                          "needed) for the SPY price feed.")
    else:
        st.sidebar.caption("🟡 REPLAY DEMO needs no setup. 🔴 LIVE needs `.env` TradeMas credentials "
                          "for the imbalance panel, and either a logged-in Bloomberg Terminal OR "
                          "`.env` Alpaca credentials (free, no Terminal needed) for the SPY price "
                          "feed — Bloomberg is used automatically when available, Alpaca otherwise.")
    if is_live:
        st.sidebar.caption("🔄 Everything refreshes in place every 1s -- never reloads the page.")

    extra_tabs = extra_tabs or []

    # `st.tabs` doesn't work here: a native `st.tabs()` panel that contains
    # ANY `st.fragment` content resets back to the first tab on every one
    # of that fragment's own auto-reruns (a known Streamlit limitation --
    # true regardless of whether `st.tabs()` itself is called inside or
    # outside the fragment; what matters is the fragment's rendered output
    # living inside a tab panel at all). A `st.radio` styled as a tab strip
    # is properly `session_state`-backed and survives fragment reruns, so
    # it's used here instead -- `key=` pins it to one widget identity
    # across reruns rather than a fresh one each time.
    tab_labels = ["🏠  Dashboard"] + [label for label, _ in extra_tabs]
    selected_tab = st.radio("View", tab_labels, horizontal=True,
                            label_visibility="collapsed", key="_active_view_tab")

    # LIVE mode is one `st.fragment(run_every=1)` -- everything (metrics,
    # regime banner, probability cards, Market Context, movers table,
    # chart, Advanced panel) refreshes every 1s. `st.fragment` reruns ONLY
    # its own function body, in place -- the surrounding page (tab
    # selection, scroll position, sidebar) never reloads or flickers.
    # REPLAY mode uses no fragment at all (no timer -- it only reruns on
    # its own widget interactions).
    def _fetch_and_render_live():
        # spy_status/imbalance_status stay None on a real reading -- only
        # set to a reason string when that feed (independently) has
        # nothing to show right now. Neither feed's absence blocks the
        # other from rendering: SPY price/probability data and NYSE
        # imbalance data come from two unrelated vendors (Bloomberg/
        # Alpaca vs. TradeMas) with no dependency on each other.
        from live_stage1_feed import NoLiveBarsYetError

        spy_status = None
        imbalance_ticks, imbalance_status = _fetch_live_imbalance_ticks()
        try:
            now, preds, features, warmed_up, bars, spy_source = _load_live_spy_state(
                engine_params, source=spy_source_mode)
        except NoLiveBarsYetError:
            now = datetime.now(pytz.timezone("US/Eastern")).replace(tzinfo=None)
            preds, features, warmed_up, bars = {}, {}, False, pd.DataFrame(columns=["time", "close"])
            spy_status = "waiting"
            spy_source = None
        except Exception as exc:
            if spy_source_mode == "alpaca":
                st.error(f"⚠️ Live SPY feed failed: {exc}\n\n"
                        "Alpaca didn't return data. Check that `ALPACA_API_KEY`/"
                        "`ALPACA_SECRET_KEY` are set in `.env` (free at https://alpaca.markets). "
                        "Switch to 🟡 REPLAY DEMO above if this persists.")
            else:
                st.error(f"⚠️ Live SPY feed failed: {exc}\n\n"
                        "Neither Bloomberg nor Alpaca returned data. Check that either (a) "
                        "Bloomberg Terminal is open and logged in on this machine (Desktop API, "
                        "localhost:8194) with `blpapi`/`xbbg` installed, or (b) `ALPACA_API_KEY`/"
                        "`ALPACA_SECRET_KEY` are set in `.env` (free at https://alpaca.markets). "
                        "Switch to 🟡 REPLAY DEMO above if this persists.")
            now = datetime.now(pytz.timezone("US/Eastern")).replace(tzinfo=None)
            preds, features, warmed_up, bars = {}, {}, False, pd.DataFrame(columns=["time", "close"])
            spy_status = "error"
            spy_source = None
        source_label = {"bloomberg": "Bloomberg", "alpaca": "Alpaca (IEX)", "massive": "Massive"}.get(spy_source)
        session_label = now.strftime("%H:%M:%S EST (live)") + (f" · {source_label}" if source_label else "")
        agg_imbalance = _live_aggregate_imbalance(imbalance_ticks)
        movers_buy, movers_sell = _live_top_movers(imbalance_ticks)
        # Load BEFORE appending today's snapshot -- these are the
        # "strictly prior" polls the z-score is computed against (see
        # live_imbalance_zscore_intraday's docstring: the current reading
        # must never be part of its own reference stats).
        prior_imbalance_hist_df = load_live_imbalance_history(now.date())
        imb_z = live_imbalance_zscore_intraday(agg_imbalance, prior_imbalance_hist_df)
        imb_z_tod = live_imbalance_zscore_tod(agg_imbalance, prior_imbalance_hist_df, now.time())
        # Only record a snapshot when TradeMas actually returned ticks.
        # `imbalance_status` is None precisely when `imbalance_ticks` is
        # non-empty (see `_fetch_live_imbalance_ticks`) -- outside the
        # feed's active window (roughly 13:30-16:00) it returns nothing
        # and `agg_imbalance` is a placeholder 0.0, not a genuine "net
        # imbalance is zero" reading, so it's never appended to the
        # session-so-far chart history.
        if imbalance_status is None:
            append_live_imbalance_snapshot(now, agg_imbalance)
        price_df = bars[["time", "close"]] if not bars.empty else pd.DataFrame(columns=["time", "close"])
        imbalance_hist_df = load_live_imbalance_history(now.date())

        now_t = now.time()
        touch_preds = touch_predictions(touch_engine, features, now_t, TAU_BPS) if warmed_up else {}
        regime_label, regime_emoji = live_regime(touch_preds, spy_status)
        regime_color = REGIME_COLOR[regime_emoji]
        cur_vol_bps = preds.get("5min", {}).get("sigma_bps", float("nan"))

        # extra_tabs (Trading/Deploy) render OUTSIDE this fragment (see why
        # above `tabs = st.tabs(...)`), so they can't just read these as
        # plain local variables on the fragment's own timer-triggered
        # reruns -- stash the latest values in session_state instead.
        st.session_state["_dash_regime_state"] = (
            regime_label, regime_emoji, regime_color, touch_preds, agg_imbalance, imb_z)
        # _render_heavy_ui (chart/movers/Advanced panel) is ALSO rendered
        # outside this fragment -- see the call site below `st.fragment(...)`
        # for why (fullscreen-mode + fragment reruns don't mix).
        st.session_state["_dash_heavy_ui_state"] = (
            True, warmed_up, movers_buy, movers_sell, price_df, imbalance_hist_df, preds, touch_preds, features)

        with st.container():
            _render_numbers_ui(st, session_label, warmed_up, True, spy_status, imbalance_status,
                               features, cur_vol_bps, agg_imbalance, imb_z, imb_z_tod,
                               regime_label, regime_emoji, regime_color, touch_preds)

    def _render_replay_body():
        try:
            now, preds, features, day_grid, sel_day, sel_time_str = _replay_state(engine_params, st)
        except FileNotFoundError as exc:
            st.error(f"⚠️ REPLAY DEMO artifacts not found: {exc}\n\n"
                    "Generate them once with `python src/build_dashboard_demo.py`, or "
                    "switch to 🔴 LIVE STREAMING above.")
            st.stop()
        session_label = f"{sel_time_str} EST  (replay {sel_day})"
        warmed_up = True
        row = day_grid[day_grid["time"].dt.strftime("%H:%M") == sel_time_str].iloc[0]
        agg_imbalance, imb_z = row["aggregate_dollar_imbalance"], row["imbalance_zscore_intraday"]
        # imbalance_zscore_tod postdates the REPLAY DEMO artifacts -- missing
        # there until build_dashboard_demo.py is rerun, not a bug.
        imb_z_tod = row["imbalance_zscore_tod"] if "imbalance_zscore_tod" in row.index else float("nan")
        movers = load_movers()
        mv = movers[(movers["date"].dt.date == sel_day) & (movers["time"] == row["time"])]
        movers_buy = mv[mv["side"] == "BUY"].sort_values("rank")
        movers_sell = mv[mv["side"] == "SELL"].sort_values("rank")
        upto = day_grid[day_grid["time"] <= row["time"]]
        price_df = upto[["time", "close"]]
        imbalance_hist_df = upto[["time", "aggregate_dollar_imbalance"]]

        now_t = now.time()
        touch_preds = touch_predictions(touch_engine, features, now_t, TAU_BPS)
        regime_label, regime_emoji = live_regime(touch_preds, None)
        regime_color = REGIME_COLOR[regime_emoji]
        cur_vol_bps = replay_predictions(row, engine_params, ["5min"])["5min"]["sigma_bps"]

        st.session_state["_dash_regime_state"] = (
            regime_label, regime_emoji, regime_color, touch_preds, agg_imbalance, imb_z)
        st.session_state["_dash_heavy_ui_state"] = (
            False, warmed_up, movers_buy, movers_sell, price_df, imbalance_hist_df, preds, touch_preds, features)

        with st.container():
            _render_numbers_ui(st, session_label, warmed_up, False, None, None,
                               features, cur_vol_bps, agg_imbalance, imb_z, imb_z_tod,
                               regime_label, regime_emoji, regime_color, touch_preds)

    if selected_tab == tab_labels[0]:
        if is_live:
            # Only the LIVE path needs the auto-rerun timer -- and only the
            # LIVE path is safe to wrap in a fragment at all: `st.fragment`
            # forbids `st.sidebar` calls from inside it, and the REPLAY DEMO
            # branch's `_replay_state` writes its day/time picker controls to
            # the sidebar. REPLAY mode reruns normally on its own widget
            # interactions, so it doesn't need (or want) a timer anyway.
            st.fragment(run_every=1)(_fetch_and_render_live)()
        else:
            _render_replay_body()

        # _render_heavy_ui (chart/movers/Advanced panel) renders OUTSIDE
        # the fast-refreshing fragment, reading the latest fetch from
        # session_state (populated just above) instead of being called
        # from inside it. Reason: going fullscreen on the Plotly chart
        # (or any element) only works as long as that DOM node keeps
        # existing -- a `st.fragment` rerun replaces the DOM subtree it
        # owns every tick, which the browser reads as "the fullscreened
        # element vanished" and silently exits fullscreen. Placed here,
        # this section only re-renders on a genuine script rerun (tab
        # switch, mode toggle, a Deploy-tab action, ...), never on the
        # fragment's own 1s timer -- the numbers above keep updating
        # continuously; the chart stays put so it can actually be looked
        # at (including fullscreened) without being yanked back every
        # second.
        (heavy_is_live, heavy_warmed_up, heavy_movers_buy, heavy_movers_sell, heavy_price_df,
         heavy_imbalance_hist_df, heavy_preds, heavy_touch_preds, heavy_features) = st.session_state.get(
            "_dash_heavy_ui_state",
            (is_live, False, [], [], pd.DataFrame(columns=["time", "close"]),
             pd.DataFrame(columns=["time", "aggregate_dollar_imbalance"]), {}, {}, {}))
        _render_heavy_ui(st, heavy_is_live, heavy_warmed_up, heavy_movers_buy, heavy_movers_sell,
                         heavy_price_df, heavy_imbalance_hist_df, heavy_preds, heavy_touch_preds,
                         heavy_features)

    # Read back OUTSIDE the `if selected_tab == ...` above so the
    # Trading/Deploy tabs still show the most recent regime read even
    # though they're not the one driving the Dashboard tab's render on this
    # particular rerun -- `_fetch_and_render_live`/`_render_replay_body` run
    # inline at least once per view (the first time that tab is selected),
    # populating this.
    regime_label, regime_emoji, regime_color, touch_preds, agg_imbalance, imb_z = st.session_state.get(
        "_dash_regime_state", (None, "⚪", REGIME_COLOR["⚪"], {}, 0.0, float("nan")))
    for label, render_fn in extra_tabs:
        if selected_tab == label:
            render_fn(st, regime_label, regime_emoji, regime_color, touch_preds, agg_imbalance, imb_z)
