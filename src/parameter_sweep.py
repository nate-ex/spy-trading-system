"""Grid / randomized parameter sweeps over StrategyConfig, for evaluating
many strategy variants automatically.

DEVELOPMENT DATA ONLY (see backtest_signals.py) -- and DELIBERATELY not a
hypothesis test: sweeping hundreds/thousands of variants is exactly the
kind of multiple-testing exposure RESEARCH_CHARTER.md's Holm-Bonferroni
correction exists for. `results['n_variants_tested']` is carried on every
row precisely so nobody downstream forgets how many things were tried
before picking a "winner" -- see the module docstring in
`dashboard_trading.py` for how the UI surfaces this. Before any sweep
"winner" is treated as a real edge (not a lucky draw), it needs its own
pre-registered Q0XX writeup with a day-clustered bootstrap CI and a
day-swap placebo test (`stats_helpers.py` has both), same as every other
finding in this project -- this module does not do that automatically.

Performance note: each variant re-runs the full day-by-day bar simulation
(see backtest_engine.py) -- a few hundred variants complete in well under
a minute on the ~830-day development window; several thousand will take
longer. `random_search` with a capped `n_iter` is the practical choice
for interactive exploration of a large space; `grid_search` is exhaustive
and best reserved for small grids (a handful of dimensions x a handful of
values each).
"""

from __future__ import annotations

import itertools
import random
from typing import Any

import pandas as pd

from backtest_engine import run_backtest
from backtest_metrics import compute_metrics
from strategy_components import StrategyConfig

ParamSpace = dict[str, list[Any]]


def _variant_configs(base: StrategyConfig, param_grid: ParamSpace) -> list[StrategyConfig]:
    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]
    configs = []
    for i, combo in enumerate(itertools.product(*value_lists)):
        overrides = dict(zip(keys, combo))
        cfg_dict = base.to_dict()
        cfg_dict.update(overrides)
        cfg_dict["name"] = f"{base.name}_v{i}"
        configs.append(StrategyConfig.from_dict(cfg_dict))
    return configs


def _run_variants(signals: pd.DataFrame, configs: list[StrategyConfig],
                  progress_cb=None) -> pd.DataFrame:
    rows = []
    n = len(configs)
    for i, cfg in enumerate(configs):
        result = run_backtest(signals, cfg)
        metrics = compute_metrics(result)
        row = {**cfg.to_dict(), **metrics, "n_variants_tested": n}
        rows.append(row)
        if progress_cb is not None:
            progress_cb(i + 1, n)
    return pd.DataFrame(rows)


def grid_search(signals: pd.DataFrame, base_config: StrategyConfig, param_grid: ParamSpace,
                progress_cb=None) -> pd.DataFrame:
    """Exhaustive sweep over the Cartesian product of `param_grid`'s
    (field_name -> list of values). Returns one row per variant: every
    StrategyConfig field + every metric from `compute_metrics` +
    `n_variants_tested`."""
    configs = _variant_configs(base_config, param_grid)
    return _run_variants(signals, configs, progress_cb)


def random_search(signals: pd.DataFrame, base_config: StrategyConfig, param_space: ParamSpace,
                  n_iter: int = 200, random_state: int | None = None, progress_cb=None) -> pd.DataFrame:
    """Randomly samples `n_iter` combinations (with replacement across
    dimensions, without duplicate combos where the space allows) from
    `param_space` (field_name -> list of candidate values)."""
    rng = random.Random(random_state)
    keys = list(param_space.keys())
    configs = []
    seen = set()
    attempts = 0
    max_attempts = n_iter * 20
    while len(configs) < n_iter and attempts < max_attempts:
        attempts += 1
        combo = tuple(rng.choice(param_space[k]) for k in keys)
        if combo in seen and len(seen) < _space_size(param_space):
            continue
        seen.add(combo)
        overrides = dict(zip(keys, combo))
        cfg_dict = base_config.to_dict()
        cfg_dict.update(overrides)
        cfg_dict["name"] = f"{base_config.name}_r{len(configs)}"
        configs.append(StrategyConfig.from_dict(cfg_dict))
    return _run_variants(signals, configs, progress_cb)


def _space_size(param_space: ParamSpace) -> int:
    size = 1
    for v in param_space.values():
        size *= max(1, len(v))
    return size
