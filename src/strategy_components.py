"""Strategy configuration schema for the backtest framework.

A flat, JSON-serializable parameter set rather than a graph of arbitrary
callables -- every rule the framework supports (entry threshold, direction
mode, imbalance filters, exit rules, sizing) is one field. This is a
deliberate tradeoff: it covers every strategy variant asked for (see
`questions`/session request) and every field is trivially sweep-able
(grid/random search just varies field values), at the cost of being less
open-ended than a full plugin DSL. Extending it: adding a new filter/exit/
sizing RULE (not just a new parameter value) means adding one field here
plus one branch in the relevant `_passes_*`/`_check_exit`/`_size_position`
function in `backtest_engine.py` -- the day/minute simulation loop itself
never changes.

IMPORTANT: probability fields (`prob_metric`/`prob_threshold`) gate on
MAGNITUDE (dispersion) probability, never direction -- this project's own
findings are that direction is a martingale in RTH. `direction_mode="both"`
takes its direction from `direction_source` -- the imbalance level's own
sign, or the chosen z-score's sign (the only two directional features
available), never from either probability engine.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

VALID_HORIZONS = ("5min", "10min", "15min", "20min", "30min")
VALID_PROB_METRICS = ("p_touch", "p_tail")
VALID_DIRECTION_MODES = ("long_only", "short_only", "both")
VALID_DIRECTION_SOURCES = ("raw_imbalance", "zscore_sign")
VALID_ZSCORE_SOURCES = ("intraday_expanding", "time_of_day_normalized", "imbalance_price_spread")
VALID_IMBALANCE_DIRECTIONS = ("any", "positive", "negative")
VALID_IMBALANCE_STATES = ("any", "reverting", "accelerating")
VALID_SIZING_MODES = ("fixed", "confidence_weighted")


@dataclass
class StrategyConfig:
    name: str = "unnamed"

    # -- entry: probability gate --
    entry_horizon: str = "5min"           # which horizon's probability triggers entry
    prob_metric: str = "p_touch"          # "p_touch" (magnitude/touch) or "p_tail" (terminal-return)
    prob_threshold: float = 0.5           # minimum probability required to consider entering

    # -- entry: direction + imbalance filters --
    direction_mode: str = "both"          # "long_only" / "short_only" / "both" (direction_source-driven)
    direction_source: str = "raw_imbalance"  # "both" mode only: "raw_imbalance" (sign of the dollar
                                           # level) or "zscore_sign" (sign of the chosen zscore_source --
                                           # "is buying pressure unusually strong RIGHT NOW", not just net
                                           # positive). These can disagree: a large-but-typical-for-today
                                           # imbalance has raw_imbalance>0 but zscore~0/negative.
    zscore_source: str = "intraday_expanding"  # which column min_abs_zscore/direction_source="zscore_sign"
                                           # read: "intraday_expanding" (imbalance_zscore_intraday,
                                           # same-day-only std -- what existing strategies were validated
                                           # against), "time_of_day_normalized" (imbalance_zscore_tod,
                                           # cross-day time-of-day-matched std -- doesn't inflate over the
                                           # session), or "imbalance_price_spread" (imbalance's z-score
                                           # NET OF SPY's own already-realized move, both z-scored the
                                           # same cross-day way -- "is imbalance running ahead of price,"
                                           # not just "is imbalance unusually large." Dev-data correlation
                                           # check (2026-07-31): raw imbalance z-score alone has ~no
                                           # forward-return correlation, but this spread does -- see
                                           # relative_state_features.add_imbalance_price_spread's
                                           # docstring. Exploratory, NOT yet a confirmed edge.
    require_imbalance_agreement: bool = False  # in long/short_only mode, imbalance sign must agree
    min_abs_zscore: float = 0.0           # minimum |zscore_source's column| to enter
    imbalance_direction: str = "any"      # required sign of aggregate_dollar_imbalance itself
    imbalance_state: str = "any"          # "any" / "reverting" / "accelerating"

    # -- entry: time-of-day window --
    time_start: str = "14:00"
    time_end: str = "16:00"

    # -- exit rules --
    holding_period_min: float | None = None   # force a time-based exit after N minutes (None = no cap)
    profit_target_bps: float | None = None    # exit when price touches entry*(1 +/- target), direction-aware
    stop_loss_bps: float | None = None        # exit when price touches entry*(1 -/+ stop), direction-aware
    eod_exit_time: str = "16:00"              # every position force-closed at/after this time

    # -- sizing --
    sizing_mode: str = "fixed"            # "fixed" or "confidence_weighted"
    fixed_fraction: float = 1.0           # fraction of capital risked per trade (sizing_mode="fixed")
    confidence_scale: float = 1.0         # sizing_mode="confidence_weighted": frac = scale * 2*max(p-0.5, 0)
    max_position_fraction: float = 1.0    # per-trade cap regardless of sizing mode
    max_total_exposure: float = 3.0       # cap on sum of concurrently-open fractions (throttles pyramiding)

    # -- costs --
    cost_bps: float = 0.0                 # round-trip; repo convention default is 0 (see q000_backtest.py)

    def __post_init__(self):
        assert self.entry_horizon in VALID_HORIZONS, self.entry_horizon
        assert self.prob_metric in VALID_PROB_METRICS, self.prob_metric
        assert self.direction_mode in VALID_DIRECTION_MODES, self.direction_mode
        assert self.direction_source in VALID_DIRECTION_SOURCES, self.direction_source
        assert self.zscore_source in VALID_ZSCORE_SOURCES, self.zscore_source
        assert self.imbalance_direction in VALID_IMBALANCE_DIRECTIONS, self.imbalance_direction
        assert self.imbalance_state in VALID_IMBALANCE_STATES, self.imbalance_state
        assert self.sizing_mode in VALID_SIZING_MODES, self.sizing_mode

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "StrategyConfig":
        return cls.from_dict(json.loads(s))

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path

    @classmethod
    def load(cls, path: Path) -> "StrategyConfig":
        return cls.from_json(Path(path).read_text())
