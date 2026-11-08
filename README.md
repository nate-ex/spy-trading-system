# SPY Trading System

A live-trading dashboard and execution engine for SPY, driven by NYSE
closing-auction order-imbalance data: two probability engines
(Stage-1 Density Engine, Touch-Probability Engine), a strategy
backtest/parameter-sweep framework, and a live Alpaca execution loop with
an independent kill switch.

The core design rule: **live trading reuses the backtest's decision
functions verbatim.** A live tick is assembled into the exact row shape the
backtester consumes, so entries, sizing and exits are decided by the same
code in both -- no reimplementation, no drift between research and
production. Every order, paper or real, goes through the same code path
(`AlpacaBroker`, with `paper: bool` explicit at every call site), so paper
trading is a genuine dry run of the real system, not a separate simulator.

> **Data note.** The imbalance data, the fitted engine artifacts
> (`data/processed/*.json`) and the development/holdout datasets come from
> a proprietary feed and are **not included** in this repo. The code,
> tests and architecture are complete; tests that need the private
> artifacts are skipped automatically (`pytest` -> 247 passed, 3 skipped).
> Validation methodology for the underlying research is in
> [NYSE-Imbalance-Research](https://github.com/nate-ex/NYSE-Imbalance-Research).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # Alpaca keys (paper first) + imbalance feed URL/credentials
pytest src/ -q
```

To run against your own data: put raw imbalance/price data in
`data/raw/`, then `python src/build_master_dataset.py` followed by
`python src/split_dataset.py` to build the development/holdout split.

## Running it

Two processes, run separately:

```bash
# 1. The background execution loop -- leave this running.
python run_live_trading.py

# 2. The dashboard.
streamlit run trading_desk.py --server.port 8502
```

The dashboard's **Deploy** tab picks a saved strategy (`StrategyConfig`,
saved via the Trading tab's Builder), starts/stops it, and shows live
positions/P&L. It talks to the background loop via small files in
`data/live_trading/` (`command.json` in, `status.json` out) -- not
directly, so closing the browser never stops an open position from being
managed.

## Kill switch

```bash
python scripts/kill_switch.py --paper   # or --live
```

Works independent of whether the dashboard or the background loop is
alive -- talks to Alpaca directly.

## Architecture

- `src/live_trading/signal_row.py` -- assembles one live tick into the
  exact row shape `backtest_signals.build_backtest_signals()` produces,
  so `backtest_engine.py`'s actual decision functions (`_passes_filters`,
  `_entry_direction`, `_size_position`, `_check_exit`, ...) decide live
  trades exactly as they decided backtested ones -- zero reimplementation,
  zero drift.
- `src/live_trading/broker.py` -- `AlpacaBroker`, the only place that
  talks to Alpaca's Trading API.
- `src/live_trading/position_manager.py` -- live open-position bookkeeping
  + order placement on entry/exit decisions.
- `src/live_trading/live_trader.py` -- the persistent polling loop.
- `src/live_trading/ipc.py` -- the command/status file protocol.
- `src/live_trading/audit_log.py` -- append-only log of every order
  intent/result -- see its docstring for why this one file matters more
  than any other in this repo.
- `trading_desk.py` -- the dashboard: `dashboard_view.render_dashboard`
  (Dashboard tab: live probability read, imbalance z-scores) and
  `dashboard_trading.render_trading_tab` (strategy research/backtest/
  sweep/compare), plus a Deploy tab.
- `src/stage1_density_engine.py` / `src/touch_probability_engine.py` --
  the two frozen prediction engines (magnitude/dispersion probability,
  never direction -- see their own docstrings).
- `src/backtest_engine.py` / `src/strategy_components.py` -- the strategy
  simulation engine and its flat, sweep-able `StrategyConfig` schema.
- `src/data_access.py` -- sanctioned loader for
  `development.parquet`/`holdout_test.parquet`; every held-out-test
  access is logged to `logs/holdout_access_log.md`.
