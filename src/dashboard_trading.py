"""Trading tab UI -- a strategy research & backtesting environment on top
of `backtest_engine.py`/`parameter_sweep.py`. Three sub-tabs: build/run a
single strategy, sweep many variants, inspect individual trades.

DEVELOPMENT DATA ONLY, and DELIBERATELY exploratory, not a confirmed
result -- see `backtest_signals.py`/`parameter_sweep.py` docstrings. This
module surfaces that on-screen (a persistent banner + `n_variants_tested`
on every sweep row) rather than letting a big Sharpe number on a results
table read as a finding. Neither probability engine implies a direction
(Q007-Q009: direction is a martingale in RTH) -- `direction_mode="both"`
strategies take direction from the imbalance sign, not from a forecast.
"""

from __future__ import annotations

import pandas as pd

from backtest_engine import run_backtest
from backtest_metrics import compute_metrics, pnl_by_time_of_day, buy_hold_benchmark
from backtest_signals import load_or_build_signals
from parameter_sweep import grid_search, random_search
from strategy_components import (
    StrategyConfig, VALID_HORIZONS, VALID_PROB_METRICS, VALID_DIRECTION_MODES,
    VALID_DIRECTION_SOURCES, VALID_ZSCORE_SOURCES,
    VALID_IMBALANCE_DIRECTIONS, VALID_IMBALANCE_STATES, VALID_SIZING_MODES,
)
import backtest_storage as bs
import strategy_sync as ss

METRIC_COLS = ["total_pnl", "avg_pnl_per_trade", "win_rate", "n_trades", "max_drawdown",
              "sharpe", "sortino", "profit_factor", "avg_holding_minutes", "avg_exposure"]


def _init_state(st) -> None:
    if "bt_last_result" not in st.session_state:
        st.session_state.bt_last_result = None
        st.session_state.bt_last_config = None
        # Loaded from disk, not blank -- LIVE mode's auto-refresh does a
        # full browser reload every poll, which would otherwise wipe these
        # (and a next-day relaunch always would) even though the user
        # never asked to clear anything. See backtest_storage.py's
        # docstring. bt_last_result/bt_last_config stay session-only (the
        # Builder tab's explicit Save/Load-by-name already covers configs
        # deliberately kept; the currently-mid-edit form isn't meant to
        # survive a reload the way a saved comparison/sweep table is).
        st.session_state.bt_comparison = bs.load_comparison()   # name -> metrics dict
        st.session_state.bt_sweep_results = bs.load_last_sweep()
        st.session_state.bt_last_daterange = None  # (d1, d2) as of the last "Run Backtest" click --
        # kept alongside bt_last_result so the SPY buy&hold benchmark shown next to a result always
        # matches the period actually backtested, even if the date pickers have since been changed.
        st.session_state.bt_flash = None  # (level, message) set right before an st.rerun(),
        # since a message shown immediately before st.rerun() never actually
        # gets painted -- the rerun starts a fresh script execution first.


def _show_flash(st) -> None:
    flash = st.session_state.get("bt_flash")
    if flash is None:
        return
    level, message = flash
    getattr(st, level)(message)
    st.session_state.bt_flash = None


def render_trading_tab(st) -> None:
    _init_state(st)
    st.subheader("Trading — Strategy Research & Backtesting")

    try:
        signals = load_or_build_signals()
    except FileNotFoundError:
        st.info(
            "📉 This tab needs `development.parquet` (this project's historical research "
            "data), which isn't present on this machine — likely a LIVE-only deployment "
            "(see `bootstrap_dashboard_data.py`). Strategy backtesting isn't available "
            "here; the Dashboard tab's live probability/imbalance panels work independently "
            "of this data and are unaffected."
        )
        return

    st.warning(
        "⚠️ **Exploratory research only, development data only.** Sweeping many strategy "
        "variants is exactly the multiple-testing exposure `RESEARCH_CHARTER.md`'s "
        "Holm-Bonferroni correction exists for — a good-looking Sharpe here is NOT a "
        "confirmed edge. Before trusting any result: (1) note `n_variants_tested`, "
        "(2) run a day-clustered bootstrap CI + day-swap placebo (`stats_helpers.py`) on "
        "the specific candidate, (3) write it up as a pre-registered `Q0XX` question, "
        "(4) only then consider a single held-out confirmation. This project's prior "
        "finding (Q008) is that no monetizable directional edge exists with a linear "
        "instrument — a screen full of positive-Sharpe rows here is the expected "
        "signature of overfitting a martingale, not evidence that finding was wrong."
    )

    min_date, max_date = signals["date"].min().date(), signals["date"].max().date()

    build_tab, sweep_tab, trades_tab, compare_tab = st.tabs(
        ["🛠️ Strategy Builder", "🔍 Parameter Sweep", "📋 Trade Inspector", "⚖️ Compare"])

    with build_tab:
        _render_builder(st, signals, min_date, max_date)
    with sweep_tab:
        _render_sweep(st, signals, min_date, max_date)
    with trades_tab:
        _render_trade_inspector(st)
    with compare_tab:
        _render_comparison(st)


def _date_range_picker(st, min_date, max_date, key_prefix: str):
    d1, d2 = st.date_input("Date range", value=(min_date, max_date), min_value=min_date,
                           max_value=max_date, key=f"{key_prefix}_daterange")
    if isinstance(d1, tuple):  # streamlit sometimes returns a tuple before both dates picked
        d1, d2 = d1
    return d1, d2


def _filter_by_date(signals: pd.DataFrame, d1, d2) -> pd.DataFrame:
    dates = signals["date"].dt.date
    return signals[(dates >= d1) & (dates <= d2)]


def _config_form(st, key_prefix: str, defaults: StrategyConfig | None = None) -> StrategyConfig:
    d = defaults or StrategyConfig()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Entry — probability gate**")
        entry_horizon = st.selectbox("Horizon", VALID_HORIZONS, index=VALID_HORIZONS.index(d.entry_horizon),
                                     key=f"{key_prefix}_horizon")
        prob_metric = st.selectbox("Probability metric", VALID_PROB_METRICS,
                                   index=VALID_PROB_METRICS.index(d.prob_metric), key=f"{key_prefix}_probmetric",
                                   help="p_touch = touch/first-passage (primary dashboard cards). "
                                       "p_tail = terminal-return (Advanced panel engine).")
        prob_threshold = st.slider("Probability threshold", 0.0, 1.0, d.prob_threshold, 0.01,
                                   key=f"{key_prefix}_probthresh",
                                   help="With Sizing = confidence_weighted below, size is "
                                       "2*max(p-0.5, 0) -- a threshold set below 0.5 will let rows "
                                       "pass this filter but still size to exactly 0 (no trade "
                                       "opens), since confidence only ramps up starting at p=0.5.")
        st.markdown("**Time window**")
        time_start = st.text_input("Start (HH:MM)", d.time_start, key=f"{key_prefix}_tstart")
        time_end = st.text_input("End (HH:MM)", d.time_end, key=f"{key_prefix}_tend")

    with c2:
        st.markdown("**Direction & imbalance filters**")
        direction_mode = st.selectbox("Direction", VALID_DIRECTION_MODES,
                                      index=VALID_DIRECTION_MODES.index(d.direction_mode),
                                      key=f"{key_prefix}_dirmode",
                                      help="'both' takes direction from Direction source below — "
                                          "neither probability engine implies a direction.")
        direction_source = st.selectbox(
            "Direction source ('both' mode only)", VALID_DIRECTION_SOURCES,
            index=VALID_DIRECTION_SOURCES.index(d.direction_source),
            key=f"{key_prefix}_dirsource",
            help="raw_imbalance: sign of the dollar level (was there net buying at all). "
                "zscore_sign: sign of the z-score below (is buying pressure unusually strong "
                "RIGHT NOW vs. its own baseline). These can disagree — a large-but-typical-for-"
                "today imbalance reads raw_imbalance>0 but zscore~0/negative.")
        zscore_source = st.selectbox(
            "Z-score source", VALID_ZSCORE_SOURCES,
            index=VALID_ZSCORE_SOURCES.index(d.zscore_source),
            key=f"{key_prefix}_zsource",
            help="intraday_expanding: same-day-only std (imbalance_zscore_intraday) — what "
                "First_Test-era strategies were validated against, but its denominator widens "
                "over the session if imbalance has real time-of-day structure. "
                "time_of_day_normalized: cross-day, same-clock-time-matched std "
                "(imbalance_zscore_tod) instead — doesn't inflate within a session.")
        require_agreement = st.checkbox("Require imbalance to agree with direction", d.require_imbalance_agreement,
                                        key=f"{key_prefix}_reqagree")
        min_abs_zscore = st.number_input("Min |z-score| (per Z-score source above)", 0.0, 10.0,
                                         d.min_abs_zscore, 0.1, key=f"{key_prefix}_minz")
        imbalance_direction = st.selectbox("Required imbalance sign", VALID_IMBALANCE_DIRECTIONS,
                                           index=VALID_IMBALANCE_DIRECTIONS.index(d.imbalance_direction),
                                           key=f"{key_prefix}_imbdir")
        imbalance_state = st.selectbox("Imbalance state", VALID_IMBALANCE_STATES,
                                       index=VALID_IMBALANCE_STATES.index(d.imbalance_state),
                                       key=f"{key_prefix}_imbstate",
                                       help="reverting = magnitude shrinking; accelerating = magnitude "
                                           "growing at an increasing rate.")

    with c3:
        st.markdown("**Exit rules**")
        use_pt = st.checkbox("Profit target", d.profit_target_bps is not None, key=f"{key_prefix}_usept")
        profit_target_bps = st.number_input("Profit target (bps)", 1.0, 500.0,
                                            d.profit_target_bps or 10.0, 1.0,
                                            key=f"{key_prefix}_ptbps", disabled=not use_pt)
        use_sl = st.checkbox("Stop loss", d.stop_loss_bps is not None, key=f"{key_prefix}_usesl")
        stop_loss_bps = st.number_input("Stop loss (bps)", 1.0, 500.0,
                                        d.stop_loss_bps or 10.0, 1.0,
                                        key=f"{key_prefix}_slbps", disabled=not use_sl)
        use_hp = st.checkbox("Max holding period", d.holding_period_min is not None, key=f"{key_prefix}_usehp")
        holding_period_min = st.number_input("Holding period (min)", 1.0, 120.0,
                                             d.holding_period_min or 15.0, 1.0,
                                             key=f"{key_prefix}_hpmin", disabled=not use_hp)
        eod_exit_time = st.text_input("EOD force-exit (HH:MM)", d.eod_exit_time, key=f"{key_prefix}_eod")

        st.markdown("**Sizing & cost**")
        sizing_mode = st.selectbox("Sizing", VALID_SIZING_MODES, index=VALID_SIZING_MODES.index(d.sizing_mode),
                                   key=f"{key_prefix}_sizing")
        fixed_fraction = st.number_input("Fixed fraction", 0.01, 5.0, d.fixed_fraction, 0.05,
                                         key=f"{key_prefix}_fixedfrac")
        confidence_scale = st.number_input("Confidence scale", 0.01, 5.0, d.confidence_scale, 0.05,
                                           key=f"{key_prefix}_confscale")
        max_position_fraction = st.number_input("Max per-trade fraction", 0.01, 5.0, d.max_position_fraction, 0.05,
                                                 key=f"{key_prefix}_maxpos")
        max_total_exposure = st.number_input("Max total exposure", 0.01, 10.0, d.max_total_exposure, 0.1,
                                             key=f"{key_prefix}_maxexp")
        cost_bps = st.number_input("Cost (bps, round-trip)", 0.0, 100.0, d.cost_bps, 1.0,
                                   key=f"{key_prefix}_cost",
                                   help="Repo convention default is 0bps (floor-algo execution, no "
                                       "commission/spread) — see q000_backtest.py.")

    return StrategyConfig(
        name=d.name, entry_horizon=entry_horizon, prob_metric=prob_metric, prob_threshold=prob_threshold,
        direction_mode=direction_mode, direction_source=direction_source, zscore_source=zscore_source,
        require_imbalance_agreement=require_agreement,
        min_abs_zscore=min_abs_zscore, imbalance_direction=imbalance_direction, imbalance_state=imbalance_state,
        time_start=time_start, time_end=time_end,
        holding_period_min=holding_period_min if use_hp else None,
        profit_target_bps=profit_target_bps if use_pt else None,
        stop_loss_bps=stop_loss_bps if use_sl else None,
        eod_exit_time=eod_exit_time, sizing_mode=sizing_mode, fixed_fraction=fixed_fraction,
        confidence_scale=confidence_scale, max_position_fraction=max_position_fraction,
        max_total_exposure=max_total_exposure, cost_bps=cost_bps,
    )


def _render_builder(st, signals, min_date, max_date) -> None:
    _show_flash(st)
    saved = bs.list_saved_configs()
    load_col, name_col = st.columns([2, 2])
    with load_col:
        to_load = st.selectbox("Load saved strategy", ["(new)"] + saved, key="builder_load")
    defaults = bs.load_config(to_load) if to_load != "(new)" else StrategyConfig()
    with name_col:
        strategy_name = st.text_input("Strategy name", defaults.name, key="builder_name")

    d1, d2 = _date_range_picker(st, min_date, max_date, "builder")

    cfg = _config_form(st, "builder", defaults)
    cfg.name = strategy_name

    run_col, save_col, delete_col = st.columns(3)
    if run_col.button("▶️ Run Backtest", type="primary", key="builder_run"):
        sub = _filter_by_date(signals, d1, d2)
        result = run_backtest(sub, cfg)
        st.session_state.bt_last_result = result
        st.session_state.bt_last_config = cfg
        st.session_state.bt_last_daterange = (d1, d2)
    if save_col.button("💾 Save Strategy", key="builder_save"):
        path = bs.save_config(cfg)
        ok, detail = ss.sync_dir(bs.CONFIGS_DIR, f"Save strategy: {cfg.name}")
        if ok:
            st.success(f"Saved to {path.name} ({detail}) — will show up on other machines next launch.")
        else:
            st.warning(f"Saved to {path.name} on this machine, but {detail}.")
    if delete_col.button("🗑️ Delete Saved Strategy", key="builder_delete", disabled=(to_load == "(new)"),
                         help="Deletes the strategy currently loaded above (via 'Load saved strategy')."
                             " Disabled when nothing saved is loaded."):
        deleted_name = to_load
        bs.delete_config(deleted_name)
        ok, detail = ss.sync_dir(bs.CONFIGS_DIR, f"Delete strategy: {deleted_name}")
        if ok:
            st.session_state.bt_flash = ("success", f"Deleted '{deleted_name}' ({detail}) — "
                                         "will disappear on other machines next launch.")
        else:
            st.session_state.bt_flash = ("warning", f"Deleted '{deleted_name}' on this machine, but {detail}.")
        st.rerun()

    result = st.session_state.bt_last_result
    if result is None or st.session_state.bt_last_config is None:
        st.caption("Run a backtest to see results.")
        return
    if st.session_state.bt_last_config.name != cfg.name and result is not None:
        pass  # stale-result note not needed; metrics below always reflect the last RUN, not the live form

    _render_result(st, result, key_prefix="builder")

    add_col, bench_col = st.columns(2)
    if add_col.button("➕ Add to comparison", key="builder_addcompare"):
        m = compute_metrics(result)
        st.session_state.bt_comparison[result.config.name] = m
        bs.save_comparison(st.session_state.bt_comparison)
        st.success(f"Added '{result.config.name}' to Compare tab ({len(st.session_state.bt_comparison)} total).")

    bench_d1, bench_d2 = st.session_state.bt_last_daterange
    bench = buy_hold_benchmark(signals, bench_d1, bench_d2)
    st.markdown(f"**vs. SPY Buy & Hold ({bench_d1} to {bench_d2}, 14:00–16:00 ET, same period)**")
    strat_m = compute_metrics(result)
    vb1, vb2, vb3 = st.columns(3)
    vb1.metric("Strategy Total P&L", f"{strat_m['total_pnl']:+.2%}")
    vb2.metric("SPY Buy & Hold Total P&L", f"{bench['total_pnl']:+.2%}")
    vb3.metric("Strategy Sharpe vs. Buy & Hold", f"{strat_m['sharpe']:.2f} vs. {bench['sharpe']:.2f}")
    if bench_col.button("➕ Add SPY Buy & Hold to comparison", key="builder_addbenchmark"):
        bench_name = f"SPY Buy&Hold {bench_d1}→{bench_d2}"
        st.session_state.bt_comparison[bench_name] = bench
        bs.save_comparison(st.session_state.bt_comparison)
        st.success(f"Added '{bench_name}' to Compare tab.")


def _render_result(st, result, key_prefix: str) -> None:
    import plotly.graph_objects as go

    metrics = compute_metrics(result)
    st.markdown(f"**Results — {result.config.name}**")
    m = st.columns(5)
    m[0].metric("Total P&L", f"{metrics['total_pnl']:+.2%}")
    m[1].metric("Win Rate", f"{metrics['win_rate']:.1%}" if pd.notna(metrics['win_rate']) else "n/a")
    m[2].metric("# Trades", f"{metrics['n_trades']:,}")
    m[3].metric("Sharpe", f"{metrics['sharpe']:.2f}")
    m[4].metric("Max Drawdown", f"{metrics['max_drawdown']:+.2%}")
    m2 = st.columns(5)
    m2[0].metric("Sortino", f"{metrics['sortino']:.2f}")
    m2[1].metric("Profit Factor", f"{metrics['profit_factor']:.2f}" if metrics["profit_factor"] not in (float("inf"),) else "∞")
    m2[2].metric("Avg P&L/Trade", f"{metrics['avg_pnl_per_trade']:+.4%}" if pd.notna(metrics['avg_pnl_per_trade']) else "n/a")
    m2[3].metric("Avg Holding Time", f"{metrics['avg_holding_minutes']:.1f} min" if pd.notna(metrics['avg_holding_minutes']) else "n/a")
    m2[4].metric("Avg Exposure", f"{metrics['avg_exposure']:.2f}x")

    if result.trades.empty:
        st.info("No trades triggered for this configuration/date range.")
        return

    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown("**Equity curve**")
        eq = result.equity
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq["time"], y=eq["equity"], mode="lines", line=dict(color="#1f77b4", width=1.5)))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), template="plotly_white",
                          yaxis=dict(title="cumulative P&L (fraction of capital)"))
        st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_equity")
    with ec2:
        st.markdown("**Distribution of trade returns**")
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=result.trades["pnl"], nbinsx=40, marker_color="#1f77b4"))
        fig2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), template="plotly_white",
                           xaxis=dict(title="P&L per trade"))
        st.plotly_chart(fig2, width="stretch", key=f"{key_prefix}_dist")

    st.markdown("**P&L by time of day**")
    tod = pnl_by_time_of_day(result.trades)
    fig3 = go.Figure()
    colors = ["#12703a" if v >= 0 else "#a83232" for v in tod["total_pnl"]]
    fig3.add_trace(go.Bar(x=tod["bucket"], y=tod["total_pnl"], marker_color=colors))
    fig3.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), template="plotly_white",
                       xaxis=dict(title="entry time bucket"), yaxis=dict(title="total P&L"))
    st.plotly_chart(fig3, width="stretch", key=f"{key_prefix}_tod")


def _parse_num_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _render_sweep(st, signals, min_date, max_date) -> None:
    st.markdown("Sweep many variants of a base strategy. Fields left as a single value are held fixed.")
    d1, d2 = _date_range_picker(st, min_date, max_date, "sweep")
    base_name = st.text_input("Base strategy name", "sweep", key="sweep_name")

    c1, c2, c3 = st.columns(3)
    with c1:
        horizons = st.multiselect("Horizons", VALID_HORIZONS, default=["5min"], key="sweep_horizons")
        prob_metrics = st.multiselect("Probability metrics", VALID_PROB_METRICS, default=["p_touch"], key="sweep_probmetrics")
        prob_thresholds = _parse_num_list(st.text_input("Probability thresholds (comma-separated)", "0.5, 0.6, 0.7", key="sweep_probthresh"))
        directions = st.multiselect("Direction modes", VALID_DIRECTION_MODES, default=["both"], key="sweep_dirmodes")
        direction_sources = st.multiselect("Direction sources ('both' mode only)", VALID_DIRECTION_SOURCES,
                                           default=["raw_imbalance"], key="sweep_dirsources")
    with c2:
        zscore_sources = st.multiselect("Z-score sources", VALID_ZSCORE_SOURCES,
                                        default=["intraday_expanding"], key="sweep_zsources")
        min_zscores = _parse_num_list(st.text_input("Min |z-score| values", "0, 1.0", key="sweep_minz"))
        imb_directions = st.multiselect("Imbalance sign", VALID_IMBALANCE_DIRECTIONS, default=["any"], key="sweep_imbdir")
        imb_states = st.multiselect("Imbalance state", VALID_IMBALANCE_STATES, default=["any"], key="sweep_imbstate")
        holding_periods = st.text_input("Holding periods, min (comma-separated, blank = no cap)", "5, 15, 30", key="sweep_hp")
    with c3:
        profit_targets = st.text_input("Profit targets, bps (comma-separated, blank = none)", "10, 20", key="sweep_pt")
        stop_losses = st.text_input("Stop losses, bps (comma-separated, blank = none)", "10, 20", key="sweep_sl")
        sizing_modes = st.multiselect("Sizing modes", VALID_SIZING_MODES, default=["fixed"], key="sweep_sizing")
        fixed_fractions = _parse_num_list(st.text_input("Fixed fractions", "1.0", key="sweep_fixedfrac"))

    mode = st.radio("Search mode", ["Grid (exhaustive)", "Random"], horizontal=True, key="sweep_mode")
    n_iter = st.number_input("Random search: # variants", 5, 5000, 100, 5, key="sweep_niter") if mode == "Random" else None

    param_space = {
        "entry_horizon": horizons or ["5min"],
        "prob_metric": prob_metrics or ["p_touch"],
        "prob_threshold": prob_thresholds or [0.5],
        "direction_mode": directions or ["both"],
        "direction_source": direction_sources or ["raw_imbalance"],
        "zscore_source": zscore_sources or ["intraday_expanding"],
        "min_abs_zscore": min_zscores or [0.0],
        "imbalance_direction": imb_directions or ["any"],
        "imbalance_state": imb_states or ["any"],
        "holding_period_min": [None if not s.strip() else float(s) for s in holding_periods.split(",")] if holding_periods.strip() else [None],
        "profit_target_bps": [None if not s.strip() else float(s) for s in profit_targets.split(",")] if profit_targets.strip() else [None],
        "stop_loss_bps": [None if not s.strip() else float(s) for s in stop_losses.split(",")] if stop_losses.strip() else [None],
        "sizing_mode": sizing_modes or ["fixed"],
        "fixed_fraction": fixed_fractions or [1.0],
    }
    n_grid = 1
    for v in param_space.values():
        n_grid *= max(1, len(v))
    st.caption(f"Grid mode would run **{n_grid:,}** variants. Runs take roughly 1s/variant — plan accordingly.")

    if st.button("🚀 Run Sweep", type="primary", key="sweep_run"):
        sub = _filter_by_date(signals, d1, d2)
        base = StrategyConfig(name=base_name)
        progress = st.progress(0.0, text="Running sweep...")
        def _cb(i, n):
            progress.progress(i / n, text=f"Running sweep... {i}/{n}")
        if mode == "Grid (exhaustive)":
            out = grid_search(sub, base, param_space, progress_cb=_cb)
        else:
            out = random_search(sub, base, param_space, n_iter=int(n_iter), random_state=42, progress_cb=_cb)
        progress.empty()
        st.session_state.bt_sweep_results = out
        bs.save_last_sweep(out)
        st.success(f"Completed {len(out)} backtests.")

    named_sweeps = bs.list_named_sweeps()
    if named_sweeps:
        load_col, btn_col = st.columns([3, 1])
        to_load = load_col.selectbox("...or reload a previously saved sweep", ["(none)"] + named_sweeps,
                                     key="sweep_loadname")
        if btn_col.button("📂 Load", key="sweep_load_btn") and to_load != "(none)":
            loaded = bs.load_sweep_results(to_load)
            st.session_state.bt_sweep_results = loaded
            bs.save_last_sweep(loaded)
            st.success(f"Loaded '{to_load}' ({len(loaded)} rows).")

    out = st.session_state.bt_sweep_results
    if out is None or out.empty:
        st.caption("Run a sweep to see results.")
        return

    st.caption(f"⚠️ {out['n_variants_tested'].iloc[0]} variants tested this sweep — see the multiple-testing "
              "warning above before treating any row as a finding.")
    sort_col = st.selectbox("Sort by", METRIC_COLS, index=METRIC_COLS.index("sharpe"), key="sweep_sortby")
    ascending = st.checkbox("Ascending", False, key="sweep_ascending")
    display_cols = ["name", "entry_horizon", "prob_metric", "prob_threshold", "direction_mode",
                    "direction_source", "zscore_source",
                    "min_abs_zscore", "imbalance_direction", "imbalance_state", "holding_period_min",
                    "profit_target_bps", "stop_loss_bps", "sizing_mode", "fixed_fraction"] + METRIC_COLS
    sorted_out = out.sort_values(sort_col, ascending=ascending)
    st.dataframe(sorted_out[display_cols], hide_index=True, width="stretch", height=400)

    st.download_button("⬇️ Export sweep results (CSV)", sorted_out.to_csv(index=False),
                       file_name=f"{base_name}_sweep.csv", key="sweep_export")

    save_name = st.text_input("Save this sweep as", base_name, key="sweep_savename")
    if st.button("💾 Save sweep results", key="sweep_save"):
        path = bs.save_sweep_results(out, save_name)
        st.success(f"Saved to {path.name}")

    pick = st.multiselect("Add row(s) to Compare tab (by index)", sorted_out.index.tolist(),
                          format_func=lambda i: sorted_out.loc[i, "name"], key="sweep_addcompare")
    if st.button("➕ Add selected to comparison", key="sweep_addcompare_btn") and pick:
        for i in pick:
            row = sorted_out.loc[i]
            st.session_state.bt_comparison[row["name"]] = {c: row[c] for c in METRIC_COLS}
        bs.save_comparison(st.session_state.bt_comparison)
        st.success(f"Added {len(pick)} strategies to Compare tab.")


def _render_trade_inspector(st) -> None:
    result = st.session_state.bt_last_result
    if result is None or result.trades.empty:
        st.caption("Run a backtest in the Strategy Builder tab to inspect its trades.")
        return

    trades = result.trades.reset_index(drop=True)
    st.markdown(f"**{len(trades)} trades — {result.config.name}**")

    import plotly.graph_objects as go
    fig = go.Figure()
    colors = ["#12703a" if p >= 0 else "#a83232" for p in trades["pnl"]]
    fig.add_trace(go.Scatter(x=trades["entry_time"], y=trades["pnl"], mode="markers",
                             marker=dict(color=colors, size=7,
                                        line=dict(width=1, color="rgba(0,0,0,0.3)")),
                             text=trades["exit_reason"], hovertemplate="%{x}<br>P&L: %{y:.4f}<br>%{text}<extra></extra>"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), template="plotly_white",
                      yaxis=dict(title="trade P&L"), xaxis=dict(title="entry time"))
    st.plotly_chart(fig, width="stretch", key="inspector_timeline")

    st.dataframe(trades[["entry_time", "exit_time", "direction", "fraction", "entry_price", "exit_price",
                        "pnl", "exit_reason", "holding_minutes"]], hide_index=True, width="stretch", height=300)

    idx = st.number_input("Inspect trade #", 0, len(trades) - 1, 0, key="inspector_idx")
    t = trades.iloc[int(idx)]
    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown("**Why entered**")
        st.info(t["entry_reason"])
        st.write(f"Direction: {'LONG' if t['direction'] == 1 else 'SHORT'} · Fraction: {t['fraction']:.2f} · "
                f"Entry: {t['entry_time']} @ {t['entry_price']:.2f}")
    with ic2:
        st.markdown("**Why exited**")
        st.info(f"Reason: {t['exit_reason']}")
        st.write(f"Exit: {t['exit_time']} @ {t['exit_price']:.2f} · Held {t['holding_minutes']:.1f} min · "
                f"P&L: {t['pnl']:+.4%}")

    st.download_button("⬇️ Export trades (CSV)", trades.to_csv(index=False),
                       file_name=f"{result.config.name}_trades.csv", key="inspector_export")


def _render_comparison(st) -> None:
    comp = st.session_state.bt_comparison
    if not comp:
        st.caption("Add strategies from the Builder or Sweep tabs to compare them here.")
        return
    df = pd.DataFrame(comp).T
    df.index.name = "strategy"
    sort_col = st.selectbox("Sort by", METRIC_COLS, index=METRIC_COLS.index("sharpe"), key="compare_sortby")
    st.dataframe(df.sort_values(sort_col, ascending=False), width="stretch", height=350)
    if st.button("🗑️ Clear comparison", key="compare_clear"):
        st.session_state.bt_comparison = {}
        bs.save_comparison({})
        st.rerun()
