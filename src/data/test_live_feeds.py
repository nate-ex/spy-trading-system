"""Unit tests for the live feed parsers and replay streamer (Part 2)."""

from __future__ import annotations

import asyncio
import sys
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd
import pytest
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_feeds import (  # noqa: E402
    EST,
    AlpacaBarFeed,
    BloombergBarFeed,
    ImbalanceTick,
    MassiveImbalanceFeed,
    MassiveSpyBarFeed,
    PriceBar,
    ReplayFeed,
    TradeMasImbalanceFeed,
    TradeMasRussellImbalanceFeed,
    _bloomberg_terminal_reachable,
    _ensure_pandas,
    _normalize_time_to_eastern,
    _require_bloomberg_terminal,
    aggregate_dollar_imbalance,
    eastern_today,
    fetch_bloomberg_vix_close_lagged,
    imbalance_zscores,
    latest_per_symbol,
    merge_into_queue,
    most_recent_close_before,
    normalize_bdh_frame,
    top_movers,
)


def _run(coro):
    return asyncio.run(coro)


async def _drain(gen, limit=1000):
    out = []
    async for item in gen:
        out.append(item)
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------
# MassiveImbalanceFeed.parse_message
# --------------------------------------------------------------------------
def test_parse_message_shares_and_side_derives_net_dollar():
    msg = {"sym": "AAPL", "timestamp": "2026-07-21 14:05:00", "imbalance": 10000,
           "side": "BUY", "ref_price": 200.0}
    tick = MassiveImbalanceFeed.parse_message(msg)
    assert tick.symbol == "AAPL"
    assert tick.side == 1
    assert tick.direction == "BUY"
    assert tick.reference_price == 200.0
    assert tick.net_imbalance_dollars == pytest.approx(10000 * 1 * 200.0)


def test_parse_message_explicit_net_imbalance_field():
    msg = {"symbol": "MSFT", "t": "2026-07-21 14:06:00", "net_imbalance": -50000.0,
           "reference_price": 410.5}
    tick = MassiveImbalanceFeed.parse_message(msg)
    assert tick.symbol == "MSFT"
    assert tick.net_imbalance_dollars == -50000.0
    assert tick.side == -1
    assert tick.direction == "SELL"


def test_parse_message_sell_side_negative_shares_flow():
    msg = {"sym": "XOM", "timestamp": "2026-07-21 14:10:00", "imbalance": 5000,
           "side": "S", "ref_price": 110.0}
    tick = MassiveImbalanceFeed.parse_message(msg)
    assert tick.side == -1
    assert tick.net_imbalance_dollars == pytest.approx(5000 * -1 * 110.0)


def test_parse_message_epoch_millis_timestamp():
    epoch_ms = int(datetime(2026, 7, 21, 18, 30, tzinfo=pytz.utc).timestamp() * 1000)
    msg = {"sym": "SPY", "timestamp": epoch_ms, "net_imbalance": 1000.0, "ref_price": 550.0}
    tick = MassiveImbalanceFeed.parse_message(msg)
    assert tick.timestamp.astimezone(EST).hour == 14  # 18:30 UTC == 14:30 EST (summer)


def test_parse_message_rejects_status_and_missing_symbol():
    assert MassiveImbalanceFeed.parse_message({"ev": "status", "message": "connected"}) is None
    assert MassiveImbalanceFeed.parse_message({"timestamp": "2026-07-21 14:00:00"}) is None


def test_parse_message_rejects_unparseable_timestamp():
    assert MassiveImbalanceFeed.parse_message({"sym": "AAPL", "timestamp": "not-a-time"}) is None


def test_start_time_gate_filters_pre_1330_ticks():
    feed = MassiveImbalanceFeed(symbols=["AAPL"])
    early = ImbalanceTick(timestamp=EST.localize(datetime(2026, 7, 21, 13, 0)), symbol="AAPL",
                          net_imbalance_dollars=100.0, side=1, reference_price=1.0)
    on_time = ImbalanceTick(timestamp=EST.localize(datetime(2026, 7, 21, 13, 30)), symbol="AAPL",
                            net_imbalance_dollars=100.0, side=1, reference_price=1.0)
    assert feed._after_start(early) is False
    assert feed._after_start(on_time) is True


# --------------------------------------------------------------------------
# ReplayFeed
# --------------------------------------------------------------------------
def _sample_imbalance_df():
    return pd.DataFrame({
        "SaleTime": pd.to_datetime(["2026-07-21 14:00:00", "2026-07-21 14:01:00"]),
        "Symbol": ["AAPL", "MSFT"],
        "Imbalance": [1000, 2000],
        "Side": [1, -1],
        "LastSale": [200.0, 400.0],
        "dollar_imbalance": [1000 * 1 * 200.0, 2000 * -1 * 400.0],
    })


def _sample_price_df():
    return pd.DataFrame({
        "time": pd.to_datetime(["2026-07-21 14:00:30", "2026-07-21 14:01:30"]),
        "symbol": ["SPY", "SPY"],
        "open": [550.0, 550.5], "high": [550.2, 550.7],
        "low": [549.9, 550.3], "close": [550.1, 550.6], "volume": [1000, 1200],
    })


def test_replay_feed_merges_and_orders_events_chronologically():
    feed = ReplayFeed(_sample_imbalance_df(), _sample_price_df(), speed=0.0)
    events = _run(_drain(feed.stream()))
    assert len(events) == 4
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["ImbalanceTick", "PriceBar", "ImbalanceTick", "PriceBar"]
    timestamps = [e.timestamp for e in events]
    assert timestamps == sorted(timestamps)


def test_replay_feed_preserves_signed_dollar_imbalance():
    feed = ReplayFeed(_sample_imbalance_df(), _sample_price_df(), speed=0.0)
    events = _run(_drain(feed.stream()))
    ticks = [e for e in events if isinstance(e, ImbalanceTick)]
    assert ticks[0].symbol == "AAPL" and ticks[0].net_imbalance_dollars > 0
    assert ticks[1].symbol == "MSFT" and ticks[1].net_imbalance_dollars < 0


def test_replay_feed_handles_empty_frames():
    feed = ReplayFeed(pd.DataFrame(), pd.DataFrame(), speed=0.0)
    events = _run(_drain(feed.stream()))
    assert events == []


# --------------------------------------------------------------------------
# merge_into_queue (non-blocking fan-in)
# --------------------------------------------------------------------------
def test_merge_into_queue_fans_in_multiple_feeds():
    feed_a = ReplayFeed(_sample_imbalance_df(), pd.DataFrame(), speed=0.0)
    feed_b = ReplayFeed(pd.DataFrame(), _sample_price_df(), speed=0.0)

    async def run():
        q: asyncio.Queue = asyncio.Queue()
        await merge_into_queue([feed_a, feed_b], q)
        out = []
        while not q.empty():
            out.append(q.get_nowait())
        return out

    out = _run(run())
    assert len(out) == 4
    assert sum(isinstance(e, ImbalanceTick) for e in out) == 2
    assert sum(isinstance(e, PriceBar) for e in out) == 2


# --------------------------------------------------------------------------
# TradeMasImbalanceFeed.parse_row
# --------------------------------------------------------------------------
def test_trademas_parse_row_camelcase_fields():
    row = {"symbol": "AAPL", "timestamp": "2026-07-21 14:05:00", "imbalance": 10000,
           "side": "BUY", "lastSale": 200.0}
    tick = TradeMasImbalanceFeed.parse_row(row)
    assert tick.symbol == "AAPL" and tick.side == 1
    assert tick.net_imbalance_dollars == pytest.approx(10000 * 200.0)


def test_trademas_parse_row_snakecase_net_imbalance():
    row = {"ticker": "MSFT", "time": "2026-07-21 14:06:00", "imbalance_dollars": -25000.0,
           "reference_price": 410.0}
    tick = TradeMasImbalanceFeed.parse_row(row)
    assert tick.symbol == "MSFT" and tick.side == -1
    assert tick.net_imbalance_dollars == -25000.0


def test_trademas_parse_row_rejects_missing_symbol():
    assert TradeMasImbalanceFeed.parse_row({"timestamp": "2026-07-21 14:00:00"}) is None


def test_trademas_parse_row_real_vendor_schema():
    # regression test: confirmed live 2026-07-28 against a populated
    # response -- the real schema uses `saleTime`, not `timestamp`/`time`/
    # `ts`, which `parse_row` didn't check for and silently dropped every
    # row (fetch_once returned [] despite a 200 response with real data).
    row = {"id": 396421, "symbol": "A", "side": 1, "imbalance": 34000,
           "pairOff": 56000, "lastSale": 140.77, "amount": 4786180.0,
           "saleTime": "2026-07-28T15:46:02.243", "flipOn": False, "deleted": None}
    tick = TradeMasImbalanceFeed.parse_row(row)
    assert tick is not None
    assert tick.symbol == "A" and tick.side == 1
    assert tick.reference_price == 140.77
    assert tick.net_imbalance_dollars == pytest.approx(4786180.0)


def test_trademas_feed_credentials_from_env(monkeypatch):
    monkeypatch.setenv("TRADEMAS_USERNAME", "user@example.com")
    monkeypatch.setenv("TRADEMAS_PASSWORD", "secret")
    feed = TradeMasImbalanceFeed()
    assert feed._auth == ("user@example.com", "secret")


def test_trademas_russell_feed_uses_its_own_base_url_by_default():
    # Regression test for a real bug: __init__'s base_url parameter used
    # to default via `= BASE_URL` (bound once, at class-DEFINITION time,
    # to the PARENT class's URL) -- a subclass overriding BASE_URL had
    # that override silently ignored unless base_url was passed
    # explicitly. Fixed via `self.BASE_URL` resolved at call time instead.
    base = TradeMasImbalanceFeed(username="u", password="p")
    russell = TradeMasRussellImbalanceFeed(username="u", password="p")
    assert base.base_url == TradeMasImbalanceFeed.BASE_URL
    assert russell.base_url == TradeMasRussellImbalanceFeed.BASE_URL
    assert russell.base_url != base.base_url


def test_trademas_russell_feed_explicit_base_url_still_overrides():
    feed = TradeMasRussellImbalanceFeed(username="u", password="p", base_url="https://example.com/custom")
    assert feed.base_url == "https://example.com/custom"


# --------------------------------------------------------------------------
# imbalance snapshot helpers
# --------------------------------------------------------------------------
def _snapshot_ticks():
    return [
        ImbalanceTick(timestamp=datetime(2026, 7, 21, 14, 0), symbol="AAPL",
                     net_imbalance_dollars=5_000_000, side=1, reference_price=200.0),
        ImbalanceTick(timestamp=datetime(2026, 7, 21, 14, 0), symbol="MSFT",
                     net_imbalance_dollars=0.0, side=0, reference_price=400.0),
        ImbalanceTick(timestamp=datetime(2026, 7, 21, 14, 0), symbol="XOM",
                     net_imbalance_dollars=-3_000_000, side=-1, reference_price=110.0),
        # stale earlier reading for AAPL -- latest_per_symbol should keep only the later one
        ImbalanceTick(timestamp=datetime(2026, 7, 21, 13, 45), symbol="AAPL",
                     net_imbalance_dollars=1_000_000, side=1, reference_price=199.0),
    ]


def test_latest_per_symbol_keeps_most_recent_tick():
    latest = latest_per_symbol(_snapshot_ticks())
    assert len(latest) == 3
    assert latest["AAPL"].net_imbalance_dollars == 5_000_000


def test_aggregate_dollar_imbalance_sums_latest_only():
    total = aggregate_dollar_imbalance(_snapshot_ticks())
    assert total == pytest.approx(5_000_000 + 0.0 - 3_000_000)


def test_imbalance_zscores_ranks_symbols():
    z = imbalance_zscores(_snapshot_ticks())
    assert z["AAPL"] > z["MSFT"] > z["XOM"]


def test_top_movers_splits_buy_and_sell():
    movers = top_movers(_snapshot_ticks(), n=5)
    assert ("AAPL", 5_000_000) in movers["buy"]
    assert ("XOM", -3_000_000) in movers["sell"]


# --------------------------------------------------------------------------
# MassiveSpyBarFeed.aggs_to_dataframe
# --------------------------------------------------------------------------
class _FakeAgg:
    def __init__(self, timestamp, open, high, low, close, volume):
        self.timestamp = timestamp
        self.open, self.high, self.low, self.close, self.volume = open, high, low, close, volume


def test_aggs_to_dataframe_converts_epoch_ms_and_sorts():
    epoch_ms_1400 = int(datetime(2026, 7, 21, 18, 0, tzinfo=pytz.utc).timestamp() * 1000)   # 14:00 EST
    epoch_ms_1401 = int(datetime(2026, 7, 21, 18, 1, tzinfo=pytz.utc).timestamp() * 1000)
    aggs = [_FakeAgg(epoch_ms_1401, 550.5, 550.7, 550.3, 550.6, 1200),
            _FakeAgg(epoch_ms_1400, 550.0, 550.2, 549.9, 550.1, 1000)]
    df = MassiveSpyBarFeed.aggs_to_dataframe(aggs)
    assert list(df["time"]) == sorted(df["time"])
    assert df.iloc[0]["time"].hour == 14 and df.iloc[0]["close"] == 550.1


def test_aggs_to_dataframe_empty_input():
    df = MassiveSpyBarFeed.aggs_to_dataframe([])
    assert df.empty
    assert list(df.columns) == ["date", "time", "open", "high", "low", "close", "volume"]


# --------------------------------------------------------------------------
# BloombergBarFeed.normalize_bdib_frame
# --------------------------------------------------------------------------
def test_normalize_bdib_frame_flat_columns():
    idx = pd.date_range("2026-07-21 14:00", periods=3, freq="1min")
    df = pd.DataFrame({"open": [1.0, 2.0, 3.0], "high": [1.1, 2.1, 3.1], "low": [0.9, 1.9, 2.9],
                       "close": [1.05, 2.05, 3.05], "volume": [100, 200, 300]}, index=idx)
    out = BloombergBarFeed.normalize_bdib_frame(df, "SPY")
    assert list(out.columns) == ["time", "open", "high", "low", "close", "volume"]
    assert len(out) == 3 and out.iloc[0]["close"] == 1.05


def test_normalize_bdib_frame_multiindex_columns():
    idx = pd.date_range("2026-07-21 14:00", periods=2, freq="1min")
    cols = pd.MultiIndex.from_product([["SPY US Equity"], ["open", "high", "low", "close", "volume"]])
    df = pd.DataFrame([[500.0, 500.5, 499.8, 500.2, 1000], [500.2, 500.6, 500.0, 500.4, 1100]],
                      index=idx, columns=cols)
    out = BloombergBarFeed.normalize_bdib_frame(df, "SPY US Equity")
    assert list(out.columns) == ["time", "open", "high", "low", "close", "volume"]
    assert len(out) == 2 and out.iloc[1]["close"] == 500.4


def test_normalize_bdib_frame_empty_or_none():
    empty_cols = ["time", "open", "high", "low", "close", "volume"]
    assert list(BloombergBarFeed.normalize_bdib_frame(None, "SPY").columns) == empty_cols
    assert list(BloombergBarFeed.normalize_bdib_frame(pd.DataFrame(), "SPY").columns) == empty_cols


class _PolarsLike:
    """Stand-in for a polars DataFrame: has no `.empty` attribute, exposes
    `.to_pandas()` -- exactly the shape that broke `normalize_bdib_frame`
    on a machine where `xbbg` returned polars instead of pandas."""

    def __init__(self, pandas_df: pd.DataFrame):
        self._pandas_df = pandas_df

    def to_pandas(self) -> pd.DataFrame:
        return self._pandas_df


def test_ensure_pandas_converts_polars_like_object():
    real = pd.DataFrame({"a": [1, 2]})
    assert _ensure_pandas(_PolarsLike(real)) is real


def test_ensure_pandas_passes_through_pandas_and_none():
    real = pd.DataFrame({"a": [1]})
    assert _ensure_pandas(real) is real
    assert _ensure_pandas(None) is None


def test_normalize_bdib_frame_handles_polars_like_input_with_datetime_index():
    idx = pd.date_range("2026-07-21 14:00", periods=2, freq="1min")
    pandas_df = pd.DataFrame({"open": [1.0, 2.0], "high": [1.1, 2.1], "low": [0.9, 1.9],
                             "close": [1.05, 2.05], "volume": [100, 200]}, index=idx)
    out = BloombergBarFeed.normalize_bdib_frame(_PolarsLike(pandas_df), "SPY")
    assert list(out.columns) == ["time", "open", "high", "low", "close", "volume"]
    assert len(out) == 2


def test_normalize_bdib_frame_uses_existing_time_column_when_present():
    # mimics a polars .to_pandas() result: RangeIndex, timestamp already a real column
    df = pd.DataFrame({"time": pd.date_range("2026-07-21 14:00", periods=2, freq="1min"),
                       "open": [1.0, 2.0], "high": [1.1, 2.1], "low": [0.9, 1.9],
                       "close": [1.05, 2.05], "volume": [100, 200]})
    out = BloombergBarFeed.normalize_bdib_frame(df, "SPY")
    assert list(out["time"]) == list(df["time"])


def test_normalize_bdib_frame_raises_on_ambiguous_schema():
    df = pd.DataFrame({"open": [1.0], "high": [1.1], "low": [0.9], "close": [1.05], "volume": [100]})
    with pytest.raises(ValueError):
        BloombergBarFeed.normalize_bdib_frame(df, "SPY")


# --------------------------------------------------------------------------
# _normalize_time_to_eastern -- the 2026-07-24 live-Bloomberg timezone bug.
#
# Round 1 fix used a median-hour-of-day heuristic (assume already-Eastern
# if the naive hour looked like typical trading hours, else assume UTC).
# That heuristic was itself wrong and reproduced the SAME live symptom
# ("no bar at/before {now} yet") because UTC trading hours (13:30-21:00)
# and Eastern trading hours (9:30-17:00) overlap for most of the session
# -- a genuinely-UTC bar at (raw) hour 16 still "looks like" a plausible
# Eastern hour. Round 2 disambiguates using the one fact known for
# certain: the true current Eastern moment (`reference_now`), by testing
# which interpretation of the latest bar is closest to (and not after) it.
# --------------------------------------------------------------------------
def test_normalize_time_to_eastern_passes_through_already_eastern_naive():
    # last bar at 14:02 ET, reference_now shortly after -- "as-is" wins
    times = pd.Series(pd.date_range("2026-07-21 14:00", periods=3, freq="1min"))
    now = pd.Timestamp("2026-07-21 14:05")
    out = _normalize_time_to_eastern(times, reference_now=now)
    assert list(out) == list(times)


def test_normalize_time_to_eastern_converts_naive_utc_like_timestamps_in_the_overlap_zone():
    # THE regression case: raw naive last bar at hour 16 (16:00) -- a value
    # that both a genuinely-Eastern 4pm bar AND a genuinely-UTC 16:00 (noon
    # Eastern) bar could produce. reference_now is the real current Eastern
    # moment: noon. Only the UTC interpretation lands at-or-before noon
    # (the as-is interpretation, 4pm, would be "in the future") -- so UTC
    # conversion must win, which the old median-hour heuristic got wrong.
    times = pd.Series(pd.date_range("2026-07-21 15:58", periods=3, freq="1min"))  # 15:58,15:59,16:00
    now = pd.Timestamp("2026-07-21 12:00")
    out = _normalize_time_to_eastern(times, reference_now=now)
    expected = pd.to_datetime(times).dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)
    assert list(out) == list(expected)
    assert out.iloc[-1] == pd.Timestamp("2026-07-21 12:00")  # 16:00 UTC == 12:00 EDT


def test_normalize_time_to_eastern_converts_tz_aware_timestamps():
    times = pd.Series(pd.date_range("2026-07-21 14:00", periods=3, freq="1min", tz="UTC"))
    out = _normalize_time_to_eastern(times)
    assert out.dt.tz is None
    # 14:00 UTC in July (EDT, UTC-4) == 10:00 ET
    assert out.iloc[0].hour == 10


def test_normalize_time_to_eastern_empty_series_passthrough():
    times = pd.Series([], dtype="datetime64[ns]")
    out = _normalize_time_to_eastern(times, reference_now=pd.Timestamp("2026-07-21 12:00"))
    assert len(out) == 0


def test_eastern_today_matches_est_localized_now():
    # regression guard: eastern_today() must track US/Eastern regardless of
    # the machine's own system timezone, not date.today() (system-local)
    import datetime as _dt
    expected = _dt.datetime.now(EST).date()
    assert eastern_today() == expected


def test_normalize_bdib_frame_corrects_utc_naive_bloomberg_timestamps_in_overlap_zone():
    # end-to-end regression test for the live bug via the actual code path
    # (normalize_bdib_frame), using the same overlap-zone scenario as above:
    # raw naive bars ending at (naive) 16:00, real current time noon ET.
    idx = pd.date_range("2026-07-21 15:58", periods=3, freq="1min")
    df = pd.DataFrame({"open": [1.0, 2.0, 3.0], "high": [1.1, 2.1, 3.1], "low": [0.9, 1.9, 2.9],
                       "close": [1.05, 2.05, 3.05], "volume": [100, 200, 300]}, index=idx)
    now = pd.Timestamp("2026-07-21 12:00")
    out = BloombergBarFeed.normalize_bdib_frame(df, "SPY", reference_now=now)
    assert out["time"].dt.tz is None
    assert out["time"].iloc[-1] == now
    assert (out["time"] <= now).all()  # the whole point: no bar left "in the future"


def test_parse_bar_row_builds_price_bar():
    row = pd.Series({"open": 500.0, "high": 500.5, "low": 499.5, "close": 500.2, "volume": 1000})
    bar = BloombergBarFeed.parse_bar_row("SPY", row, datetime(2026, 7, 21, 14, 0))
    assert bar.symbol == "SPY" and bar.close == 500.2 and bar.source == "bloomberg"


# --------------------------------------------------------------------------
# AlpacaBarFeed.bars_to_dataframe (free IEX feed, Bloomberg-free fallback)
# --------------------------------------------------------------------------
class _FakeAlpacaBar:
    def __init__(self, timestamp, open, high, low, close, volume):
        self.timestamp, self.open, self.high = timestamp, open, high
        self.low, self.close, self.volume = low, close, volume


def test_alpaca_bars_to_dataframe_converts_tz_aware_utc_and_sorts():
    ts_1400 = datetime(2026, 7, 21, 18, 0, tzinfo=pytz.utc)   # 14:00 EST
    ts_1401 = datetime(2026, 7, 21, 18, 1, tzinfo=pytz.utc)
    bars = [_FakeAlpacaBar(ts_1401, 550.5, 550.7, 550.3, 550.6, 1200),
            _FakeAlpacaBar(ts_1400, 550.0, 550.2, 549.9, 550.1, 1000)]
    df = AlpacaBarFeed.bars_to_dataframe(bars)
    assert list(df["time"]) == sorted(df["time"])
    assert df.iloc[0]["time"].hour == 14 and df.iloc[0]["close"] == 550.1
    assert df.iloc[0]["time"].tzinfo is None  # tz-naive Eastern wall-clock


def test_alpaca_bars_to_dataframe_handles_naive_timestamp_as_utc():
    # alpaca-py `Bar.timestamp` is always tz-aware in practice, but the
    # conversion should still do something sane (treat as UTC) if it ever
    # isn't, rather than silently misaligning like the original Bloomberg
    # timezone bug this project already hit once (see
    # `_normalize_time_to_eastern`'s docstring).
    naive_ts = datetime(2026, 7, 21, 18, 0)  # would be 14:00 EST if UTC
    bars = [_FakeAlpacaBar(naive_ts, 550.0, 550.2, 549.9, 550.1, 1000)]
    df = AlpacaBarFeed.bars_to_dataframe(bars)
    assert df.iloc[0]["time"].hour == 14


def test_alpaca_bars_to_dataframe_empty_input():
    df = AlpacaBarFeed.bars_to_dataframe([])
    assert df.empty
    assert list(df.columns) == ["date", "time", "open", "high", "low", "close", "volume"]


def test_alpaca_bars_to_dataframe_dict_like_input():
    ts = datetime(2026, 7, 21, 18, 0, tzinfo=pytz.utc)
    bars = [{"timestamp": ts, "open": 550.0, "high": 550.2, "low": 549.9, "close": 550.1, "volume": 1000}]
    df = AlpacaBarFeed.bars_to_dataframe(bars)
    assert df.iloc[0]["close"] == 550.1


def test_alpaca_bar_feed_client_requires_credentials(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    feed = AlpacaBarFeed(tickers=["SPY"], api_key=None, secret_key=None)
    with pytest.raises(RuntimeError, match="ALPACA_API_KEY"):
        feed._client()


def test_alpaca_bar_feed_credentials_from_env(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "fake-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "fake-secret")
    feed = AlpacaBarFeed(tickers=["SPY"])
    assert feed.api_key == "fake-key" and feed.secret_key == "fake-secret"


# --------------------------------------------------------------------------
# Bloomberg lagged VIX close (normalize_bdh_frame / most_recent_close_before)
# --------------------------------------------------------------------------
def test_normalize_bdh_frame_flat_column():
    idx = pd.date_range("2026-07-15", periods=3, freq="D")
    df = pd.DataFrame({"PX_LAST": [15.0, 16.0, 17.0]}, index=idx)
    out = normalize_bdh_frame(df)
    assert list(out.columns) == ["date", "px_last"]
    assert len(out) == 3 and out.iloc[-1]["px_last"] == 17.0


def test_normalize_bdh_frame_multiindex_column():
    idx = pd.date_range("2026-07-15", periods=2, freq="D")
    cols = pd.MultiIndex.from_tuples([("VIX Index", "PX_LAST")])
    df = pd.DataFrame([[15.5], [16.5]], index=idx, columns=cols)
    out = normalize_bdh_frame(df)
    assert list(out.columns) == ["date", "px_last"]
    assert out.iloc[0]["px_last"] == 15.5


def test_normalize_bdh_frame_empty_or_none():
    assert list(normalize_bdh_frame(None).columns) == ["date", "px_last"]
    assert list(normalize_bdh_frame(pd.DataFrame()).columns) == ["date", "px_last"]


def test_normalize_bdh_frame_handles_polars_like_input():
    idx = pd.date_range("2026-07-15", periods=2, freq="D")
    pandas_df = pd.DataFrame({"PX_LAST": [15.0, 16.0]}, index=idx)
    out = normalize_bdh_frame(_PolarsLike(pandas_df))
    assert list(out.columns) == ["date", "px_last"]
    assert len(out) == 2


def test_normalize_bdh_frame_raises_on_ambiguous_schema():
    df = pd.DataFrame({"px_last": [15.0]})
    with pytest.raises(ValueError):
        normalize_bdh_frame(df)


def test_normalize_bdh_frame_ticker_labeled_flat_column():
    # regression test for the live bug: "could not convert string to float:
    # 'VIX Index'" -- a single-ticker/single-field bdh call came back with
    # a flat column literally named "VIX Index" (by ticker, not field).
    # The values themselves are numeric; must still resolve to px_last.
    idx = pd.date_range("2026-07-15", periods=3, freq="D")
    df = pd.DataFrame({"VIX Index": [15.0, 16.0, 17.0]}, index=idx)
    out = normalize_bdh_frame(df)
    assert list(out.columns) == ["date", "px_last"]
    assert out.iloc[-1]["px_last"] == 17.0


def test_normalize_bdh_frame_prefers_numeric_column_over_ticker_label_column():
    # a remaining non-date column that is a non-numeric ticker/metadata
    # label must be skipped in favor of the actual numeric price column,
    # regardless of which one appears first.
    idx = pd.date_range("2026-07-15", periods=3, freq="D")
    df = pd.DataFrame({"ticker": ["VIX Index"] * 3, "PX_LAST": [15.0, 16.0, 17.0]}, index=idx)
    out = normalize_bdh_frame(df)
    assert list(out.columns) == ["date", "px_last"]
    assert out.iloc[-1]["px_last"] == 17.0


def test_normalize_bdh_frame_raises_clear_error_when_nothing_numeric():
    idx = pd.date_range("2026-07-15", periods=2, freq="D")
    df = pd.DataFrame({"ticker": ["VIX Index", "VIX Index"]}, index=idx)
    with pytest.raises(ValueError, match="no numeric values"):
        normalize_bdh_frame(df)


def test_most_recent_close_before_excludes_as_of_date():
    df = pd.DataFrame({"date": pd.to_datetime(["2026-07-20", "2026-07-21", "2026-07-22"]),
                       "px_last": [15.0, 16.0, 17.0]})
    val = most_recent_close_before(df, date(2026, 7, 22))
    assert val == 16.0  # 07-21's close, not 07-22's (today, excluded)


def test_most_recent_close_before_raises_when_nothing_prior():
    df = pd.DataFrame({"date": pd.to_datetime(["2026-07-22"]), "px_last": [17.0]})
    with pytest.raises(RuntimeError):
        most_recent_close_before(df, date(2026, 7, 22))


def test_fetch_bloomberg_vix_close_lagged_uses_bdh_and_filters(monkeypatch):
    class _FakeBlp:
        @staticmethod
        def bdh(tickers, flds, start_date, end_date):
            idx = pd.date_range(start_date, end_date, freq="D")
            return pd.DataFrame({flds: list(range(10, 10 + len(idx)))}, index=idx)

    import sys as _sys
    import types

    fake_xbbg = types.ModuleType("xbbg")
    fake_xbbg.blp = _FakeBlp
    monkeypatch.setitem(_sys.modules, "xbbg", fake_xbbg)
    monkeypatch.setattr("live_feeds._bloomberg_terminal_reachable", lambda timeout=1.5: True)

    val = fetch_bloomberg_vix_close_lagged(as_of=date(2026, 7, 22), lookback_days=5)
    assert val == 10 + 4  # last row strictly before 2026-07-22 in a 5-day-back range


def test_bloomberg_terminal_reachable_false_when_port_closed():
    # No real Bloomberg Terminal in the test environment -- confirms the
    # fast preflight (not xbbg's own ~50s internal retry logic) is what
    # fires, so callers fail in well under a second, not a minute.
    assert _bloomberg_terminal_reachable(timeout=0.3) is False


def test_require_bloomberg_terminal_raises_clear_error(monkeypatch):
    monkeypatch.setattr("live_feeds._bloomberg_terminal_reachable", lambda timeout=1.5: False)
    with pytest.raises(RuntimeError, match="not reachable"):
        _require_bloomberg_terminal()
