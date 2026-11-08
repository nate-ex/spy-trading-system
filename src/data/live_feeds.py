"""Unified live/replay streaming layer — MVI Live Execution Pipeline (Part 2).

Real feed handlers + one offline replay handler, all normalized onto the
same two record types (`ImbalanceTick`, `PriceBar`) and the same `async def
stream()` interface, so `src/execution/live_runner.py` never needs to know
whether it's looking at a live socket, a REST poll, or a CSV replay.

Handlers:
  - `TradeMasImbalanceFeed` — the real NYSE order-imbalance feed
    (`$IMBALANCE_FEED_URL`, HTTP Basic
    Auth). This is the primary imbalance source for the live dashboard.
    S&P 500-scoped only (~500 symbols, confirmed empirically) -- does NOT
    include Russell names, see the class's own docstring. Credentials
    come from `TRADEMAS_USERNAME`/`TRADEMAS_PASSWORD` env vars (`.env`,
    gitignored) — never hardcode them.
  - `TradeMasRussellImbalanceFeed` — Russell-scoped counterpart, added
    2026-07-31. See its docstring for current status/known gaps.
  - `MassiveImbalanceFeed`  — kept as an alternate/backup NYSE order-imbalance
    source (Massive API). WebSocket push if `MASSIVE_WS_URL` is configured,
    else REST polling. MUST be run locally with real network access to
    api.massive.com / the Massive WS gateway — this project's sandbox has
    no such access.
  - `BloombergBarFeed`      — 1-minute bars / live ticks via `xbbg`/`blpapi`.
    MUST be run on a machine with a live, logged-in Bloomberg Terminal
    (Desktop API on localhost:8194). See `fetch_spy_bloomberg.py` for the
    same constraint on the historical-fetch side of this codebase. This is
    the preferred SPY price source when available, but it is NOT required.
  - `AlpacaBarFeed`         — 1-minute bars via Alpaca's free IEX feed
    (`alpaca-py`). No Bloomberg Terminal and no paid subscription needed —
    just a free Alpaca account (paper or live) and an API key pair
    (`ALPACA_API_KEY`/`ALPACA_SECRET_KEY` env vars, `.env`, gitignored).
    This is the primary/preferred fallback when Bloomberg isn't available
    (see `live_stage1_feed.fetch_today_spy_bars`'s `source="auto"` — tries
    Bloomberg first, falls back to Alpaca on any failure). IEX is a single
    exchange (~2.5% of consolidated US equity volume) rather than the full
    SIP tape, but SPY is liquid enough that this is a reasonable proxy —
    not identical to Bloomberg/SIP-sourced bars, just close.
  - `MassiveSpyBarFeed`     — kept as a further fallback (Massive/Polygon
    API). Requires a paid Massive plan for intraday bars — the free tier is
    end-of-day only (see `fetch_spy_massive.py`), so this is not useful for
    a live dashboard on a free-tier key.
  - `ReplayFeed`            — dry-run fallback: replays recent historical
    imbalance/price CSVs tick-by-tick (respecting original inter-tick
    timing, optionally sped up), for offline testing when none of the
    above is reachable (no Terminal, no network, CI).

All streaming handlers are async generators (`async def stream(self)`),
driven with lightweight `asyncio` tasks pushing onto a shared
`asyncio.Queue` — see `merge_into_queue()` — so a slow/blocked network call
on one feed never blocks signal computation reading from the queue.

Real data (`data/raw/`) is never committed to git — same non-negotiable as
the rest of this repo (see CLAUDE.md).
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import AsyncIterator, Iterable, Optional

import pandas as pd
import pytz

EST = pytz.timezone("US/Eastern")
EARLY_INDICATION_START = time(13, 30)   # feed goes live for "usual suspects" at 13:30 EST
BLOOMBERG_DAPI_PORT = 8194


def _bloomberg_terminal_reachable(timeout: float = 1.5) -> bool:
    """Fast preflight: is anything even listening on localhost:8194 (the
    Bloomberg Desktop API port)? Confirmed live 2026-07-29: with no
    Terminal running, `xbbg`/`blpapi`'s own internal retry logic takes
    ~50s of repeated connection attempts before finally giving up -- which
    would make every "auto" dashboard poll hang for the better part of a
    minute if the Terminal is closed or crashes mid-session, before ever
    reaching the Alpaca fallback. A plain TCP connect attempt fails in well
    under a second when nothing is listening, so callers check this FIRST
    and raise immediately rather than waiting through xbbg's own slow
    retries -- this is a no-op timing-wise on a machine where the Terminal
    genuinely is running (the normal case), since the connect succeeds
    instantly there too."""
    try:
        with socket.create_connection(("127.0.0.1", BLOOMBERG_DAPI_PORT), timeout=timeout):
            return True
    except OSError:
        return False


def _require_bloomberg_terminal() -> None:
    if not _bloomberg_terminal_reachable():
        raise RuntimeError(
            "Bloomberg Terminal not reachable on localhost:8194 -- is it open "
            "and logged in on this machine?"
        )


def eastern_today() -> date:
    """Today's date in US/Eastern, independent of the machine's own system
    timezone setting -- `date.today()` uses local system time, which is a
    real risk on a machine not configured for US/Eastern (the same class
    of bug as the Bloomberg-bar-timestamp fix: a naive-local-time
    assumption). Used everywhere this module queries Bloomberg/Massive for
    "today's" bars, so a non-Eastern system clock can't silently request
    the wrong trading day."""
    return datetime.now(EST).date()

# Major S&P 500 "usual suspects" tracked from the 13:30 early-indication window.
# Best-effort curated large-cap list (mirrors the sector proxy already used in
# research/phase5_microstructure_alpha/phase5_strategies.py); not a point-in-time
# index membership file, so it is not used as a universe filter anywhere else.
USUAL_SUSPECTS = [
    "SPY", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM", "BRK B",
    "JNJ", "V", "MA", "UNH", "HD", "PG", "BAC", "CVX", "GS", "CAT",
]
DEFAULT_ETFS = ["SPY", "XLF", "XLI", "XLE"]


# --------------------------------------------------------------------------
# normalized record types
# --------------------------------------------------------------------------
@dataclass
class ImbalanceTick:
    """One NYSE order-imbalance update. Sign convention: positive
    `net_imbalance_dollars` = net BUY imbalance (matches `dollar_imbalance` =
    Imbalance*Side*LastSale in `load_imbalances.py`)."""
    timestamp: datetime
    symbol: str
    net_imbalance_dollars: float
    side: int            # +1 buy imbalance, -1 sell imbalance, 0 flat/paired
    reference_price: float
    source: str = "massive"

    @property
    def direction(self) -> str:
        return {1: "BUY", -1: "SELL", 0: "FLAT"}.get(self.side, "FLAT")


@dataclass
class PriceBar:
    """One OHLCV bar (or a degenerate 1-tick bar where open==high==low==close)."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str = "bloomberg"


# --------------------------------------------------------------------------
# Massive API — NYSE imbalance feed
# --------------------------------------------------------------------------
class MassiveImbalanceFeed:
    """Real-time NYSE order-imbalance feed via the Massive API.

    Prefers a WebSocket push subscription (`MASSIVE_WS_URL` env var set);
    falls back to REST polling (`MASSIVE_REST_BASE`, default
    `https://api.massive.com`) on a fixed interval otherwise. Both paths
    parse each update into an `ImbalanceTick` with: timestamp, symbol, net
    imbalance dollar value, imbalance direction (+1/-1), and reference
    price — the four fields the brief calls out.

    `start_time` gates emission: nothing is yielded before 13:30 EST (the
    feed's actual early-indication start), matching the desk's convention
    that Stage-1 state doesn't begin accumulating before then.
    """

    def __init__(
        self,
        symbols: Iterable[str] = USUAL_SUSPECTS,
        api_key: Optional[str] = None,
        ws_url: Optional[str] = None,
        rest_base: Optional[str] = None,
        poll_seconds: float = 2.0,
        start_time: time = EARLY_INDICATION_START,
    ):
        self.symbols = list(symbols)
        self.api_key = api_key or os.environ.get("MASSIVE_API_KEY")
        self.ws_url = ws_url or os.environ.get("MASSIVE_WS_URL")
        self.rest_base = rest_base or os.environ.get("MASSIVE_REST_BASE", "https://api.massive.com")
        self.poll_seconds = poll_seconds
        self.start_time = start_time

    # ---- parsing (pure, unit-testable without any network) ----
    @staticmethod
    def parse_message(msg: dict) -> Optional[ImbalanceTick]:
        """Parse one raw imbalance message (websocket frame or REST row) into
        an `ImbalanceTick`. Returns None for messages that aren't imbalance
        events (e.g. auth/status acks) or that fail to parse.

        Expected fields (Massive imbalance schema, either naming style):
          sym / symbol, timestamp (epoch ms or ISO8601) / t,
          imbalance / net_imbalance (signed dollars) OR
              (shares + side + ref_price) to derive it,
          side (+1/-1 or "BUY"/"SELL"), ref_price / reference_price / last_price.
        """
        if not isinstance(msg, dict) or msg.get("ev") in {"status", "auth"}:
            return None

        symbol = msg.get("sym") or msg.get("symbol") or msg.get("S")
        if not symbol:
            return None

        raw_ts = msg.get("timestamp", msg.get("t"))
        ts = MassiveImbalanceFeed._parse_timestamp(raw_ts)
        if ts is None:
            return None

        ref_price = float(msg.get("ref_price", msg.get("reference_price", msg.get("last_price", 0.0)) or 0.0))

        raw_side = msg.get("side", msg.get("imbalance_side"))
        side = MassiveImbalanceFeed._normalize_side(raw_side)

        if "net_imbalance" in msg or "imbalance_dollars" in msg:
            net = float(msg.get("net_imbalance", msg.get("imbalance_dollars")))
            if side == 0 and net != 0:
                side = 1 if net > 0 else -1
        else:
            shares = float(msg.get("imbalance", msg.get("shares", 0.0)) or 0.0)
            net = shares * side * ref_price

        return ImbalanceTick(
            timestamp=ts, symbol=str(symbol), net_imbalance_dollars=net,
            side=side, reference_price=ref_price, source="massive",
        )

    @staticmethod
    def _parse_timestamp(raw) -> Optional[datetime]:
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            # epoch millis if large, else epoch seconds
            secs = raw / 1000.0 if raw > 1e12 else float(raw)
            return datetime.fromtimestamp(secs, tz=pytz.utc).astimezone(EST)
        try:
            dt = pd.to_datetime(raw)
        except (ValueError, TypeError):
            return None
        if dt.tzinfo is None:
            dt = EST.localize(dt.to_pydatetime())
        return dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt

    @staticmethod
    def _normalize_side(raw) -> int:
        if raw is None:
            return 0
        if isinstance(raw, (int, float)):
            return 1 if raw > 0 else (-1 if raw < 0 else 0)
        s = str(raw).strip().upper()
        if s in {"B", "BUY"}:
            return 1
        if s in {"S", "SELL"}:
            return -1
        return 0

    def _after_start(self, tick: ImbalanceTick) -> bool:
        return tick.timestamp.astimezone(EST).time() >= self.start_time

    # ---- transports ----
    async def stream(self) -> AsyncIterator[ImbalanceTick]:
        if self.ws_url:
            async for tick in self._stream_ws():
                if self._after_start(tick):
                    yield tick
        else:
            async for tick in self._stream_rest_poll():
                if self._after_start(tick):
                    yield tick

    async def _stream_ws(self) -> AsyncIterator[ImbalanceTick]:
        import websockets

        subscribe = {"action": "subscribe", "params": ",".join(f"NOI.{s}" for s in self.symbols)}
        async with websockets.connect(self.ws_url) as ws:
            if self.api_key:
                await ws.send(json.dumps({"action": "auth", "params": self.api_key}))
            await ws.send(json.dumps(subscribe))
            async for raw in ws:
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                messages = payload if isinstance(payload, list) else [payload]
                for msg in messages:
                    tick = self.parse_message(msg)
                    if tick is not None:
                        yield tick

    async def _stream_rest_poll(self) -> AsyncIterator[ImbalanceTick]:
        import requests

        seen: set[tuple] = set()
        while True:
            try:
                resp = await asyncio.to_thread(
                    requests.get,
                    f"{self.rest_base}/v1/imbalances",
                    params={"tickers": ",".join(self.symbols), "apiKey": self.api_key},
                    timeout=5,
                )
                resp.raise_for_status()
                rows = resp.json().get("results", [])
            except Exception as exc:  # network hiccups shouldn't kill the poller
                print(f"[MassiveImbalanceFeed] poll failed: {exc}")
                rows = []
            for row in rows:
                tick = self.parse_message(row)
                if tick is None:
                    continue
                key = (tick.symbol, tick.timestamp, tick.net_imbalance_dollars)
                if key in seen:
                    continue
                seen.add(key)
                yield tick
            await asyncio.sleep(self.poll_seconds)


# --------------------------------------------------------------------------
# TradeMas API — primary NYSE imbalance feed (S&P 500-scoped -- see below)
# --------------------------------------------------------------------------
class TradeMasImbalanceFeed:
    """Real-time NYSE order-imbalance feed via the TradeMas API's "basic"
    endpoint (`GET $IMBALANCE_FEED_URL`,
    HTTP Basic Auth). REST-poll only (no WebSocket documented for this
    vendor).

    IMPORTANT -- universe scope (confirmed empirically 2026-07-31, cross-
    checked against a month of raw data): this "basic" endpoint reports
    on roughly S&P 500 constituents only (~500 unique symbols seen), NOT
    the full NYSE imbalance universe -- Russell 2000/3000 names are NOT
    included. `aggregate_dollar_imbalance`/anything summed from this
    feed's ticks is really "net S&P 500 imbalance," not "net NYSE
    imbalance" -- the dashboard/live desk label this explicitly to avoid
    the earlier mislabeling. See `TradeMasRussellImbalanceFeed` (this
    module) for the Russell-scoped counterpart, once its real endpoint is
    confirmed -- see that class's docstring for current status.

    Confirmed live 2026-07-28 against a populated response during trading
    hours. Real row schema: `{id, symbol, side (+1/-1), imbalance (shares),
    pairOff, lastSale, amount (= imbalance*side*lastSale, i.e. signed $
    imbalance), saleTime (ISO8601, no offset), flipOn, deleted}`. No/bad
    credentials -> 401 `{"success": false, "error": "Unauthorized", ...}`.
    `parse_row` is written defensively against the common naming variants
    seen in this codebase's other feeds (see
    `MassiveImbalanceFeed.parse_message`), plus this vendor's actual
    `saleTime` field.
    """

    BASE_URL = os.environ.get("IMBALANCE_FEED_URL", "")  # private vendor endpoint, set in .env

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None,
                 poll_seconds: float = 2.0, start_time: time = EARLY_INDICATION_START,
                 base_url: Optional[str] = None):
        # base_url defaults via self.BASE_URL (resolved per-instance, at
        # call time), NOT a `= BASE_URL` parameter default -- Python binds
        # parameter defaults once, at function-DEFINITION time, so a
        # `= BASE_URL` default would always resolve to THIS class's
        # BASE_URL even for a subclass that overrides it (e.g.
        # TradeMasRussellImbalanceFeed) -- silently ignoring the
        # subclass's own endpoint.
        self.username = username or os.environ.get("TRADEMAS_USERNAME")
        self.password = password or os.environ.get("TRADEMAS_PASSWORD")
        self.poll_seconds = poll_seconds
        self.start_time = start_time
        self.base_url = base_url or self.BASE_URL

    @property
    def _auth(self) -> tuple:
        return (self.username, self.password)

    @staticmethod
    def parse_row(row: dict) -> Optional[ImbalanceTick]:
        """Parse one TradeMas imbalance row into an `ImbalanceTick`. Accepts
        the same broad set of field-name aliases as
        `MassiveImbalanceFeed.parse_message` (symbol/sym/ticker,
        timestamp/time/ts, net_imbalance/imbalance_dollars or
        shares+side+price, side/direction, reference/last price)."""
        if not isinstance(row, dict):
            return None
        symbol = row.get("symbol") or row.get("sym") or row.get("ticker")
        if not symbol:
            return None
        raw_ts = row.get("timestamp", row.get("time", row.get("ts", row.get("saleTime"))))
        ts = MassiveImbalanceFeed._parse_timestamp(raw_ts)
        if ts is None:
            return None
        ref_price = float(row.get("referencePrice", row.get("reference_price",
                          row.get("lastSale", row.get("last_price", 0.0)))) or 0.0)
        raw_side = row.get("side", row.get("direction", row.get("imbalanceSide")))
        side = MassiveImbalanceFeed._normalize_side(raw_side)
        if "net_imbalance" in row or "imbalanceDollars" in row or "imbalance_dollars" in row:
            net = float(row.get("net_imbalance", row.get("imbalanceDollars", row.get("imbalance_dollars"))))
            if side == 0 and net != 0:
                side = 1 if net > 0 else -1
        else:
            shares = float(row.get("imbalance", row.get("shares", row.get("imbalanceShares", 0.0))) or 0.0)
            net = shares * side * ref_price
        return ImbalanceTick(timestamp=ts, symbol=str(symbol), net_imbalance_dollars=net,
                             side=side, reference_price=ref_price, source="trademas")

    def _after_start(self, tick: ImbalanceTick) -> bool:
        return tick.timestamp.astimezone(EST).time() >= self.start_time

    # ---- one-shot fetch (used directly by the Streamlit dashboard, which
    # reruns synchronously rather than driving an asyncio generator) ----
    def fetch_once(self, timeout: float = 10.0) -> list[ImbalanceTick]:
        import requests

        resp = requests.get(self.base_url, auth=self._auth, timeout=timeout)
        resp.raise_for_status()
        rows = resp.json()
        if isinstance(rows, dict):
            rows = rows.get("results", rows.get("data", []))
        ticks = [self.parse_row(r) for r in rows]
        return [t for t in ticks if t is not None]

    # ---- async streaming (used by the live execution pipeline) ----
    async def stream(self) -> AsyncIterator[ImbalanceTick]:
        seen: set[tuple] = set()
        while True:
            try:
                ticks = await asyncio.to_thread(self.fetch_once)
            except Exception as exc:  # network hiccups shouldn't kill the poller
                print(f"[TradeMasImbalanceFeed] poll failed: {exc}")
                ticks = []
            for tick in ticks:
                if not self._after_start(tick):
                    continue
                key = (tick.symbol, tick.timestamp, tick.net_imbalance_dollars)
                if key in seen:
                    continue
                seen.add(key)
                yield tick
            await asyncio.sleep(self.poll_seconds)


class TradeMasRussellImbalanceFeed(TradeMasImbalanceFeed):
    """Russell-scoped counterpart to `TradeMasImbalanceFeed` -- SCAFFOLD,
    not yet wired to a confirmed real endpoint as of 2026-07-31.

    `TradeMasImbalanceFeed`'s "basic" endpoint
    (`/api/basic/imbalance`) is confirmed S&P 500-scoped only (~500
    symbols, empirically verified against a month of raw data + the live
    feed) -- it does NOT include Russell names. Reasonable API discovery
    attempts to find a Russell-scoped sibling endpoint on the same
    TradeMas API came up empty: tried plausible sibling paths
    (`/api/russell/imbalance`, `/api/full/imbalance`, `/api/nyse/
    imbalance`, etc. -- all fall through to the same generic Blazor SPA
    catch-all page, not real routes), swagger/API-doc discovery
    (`/swagger`, `/swagger/v1/swagger.json` -- genuine 404s, so no
    swagger UI configured, or not at that path), and query parameters on
    the working endpoint (`?universe=russell`, `?index=RUT`, etc. -- all
    silently ignored, same 339-row response every time regardless of
    params).

    `BASE_URL` below is a placeholder guess (`/api/basic/imbalance/
    russell`) -- **update it** once the real endpoint/params are found
    (e.g. via the TradeMas web app's own UI + browser Network tab, since
    that Blazor app -- confirmed live at the same domain -- almost
    certainly calls the real Russell-scoped route somewhere if one
    exists). Everything else (auth, response schema, `parse_row`,
    `fetch_once`, `stream`) is inherited unchanged from
    `TradeMasImbalanceFeed` since this is the same vendor/API, just
    (presumed) a different query scope -- update `parse_row` too if the
    real endpoint's response shape turns out to differ.
    """

    BASE_URL = os.environ.get("IMBALANCE_FEED_URL", "").rstrip("/") + "/russell"  # TODO: confirm real endpoint


# --------------------------------------------------------------------------
# pure snapshot helpers (imbalance context panel — no network, no asyncio)
# --------------------------------------------------------------------------
def latest_per_symbol(ticks: Iterable[ImbalanceTick]) -> dict:
    """Most recent tick per symbol (by timestamp) from a flat tick list."""
    out: dict[str, ImbalanceTick] = {}
    for t in ticks:
        cur = out.get(t.symbol)
        if cur is None or t.timestamp >= cur.timestamp:
            out[t.symbol] = t
    return out


def aggregate_dollar_imbalance(ticks: Iterable[ImbalanceTick]) -> float:
    """Cross-sectional sum of net $ imbalance across the latest reading per
    symbol -- the snapshot analog of `load_imbalances.build_aggregate_series`."""
    latest = latest_per_symbol(ticks)
    return float(sum(t.net_imbalance_dollars for t in latest.values()))


def imbalance_zscores(ticks: Iterable[ImbalanceTick]) -> dict:
    """{symbol: z} of net $ imbalance across the latest reading per symbol."""
    import numpy as np

    latest = latest_per_symbol(ticks)
    if len(latest) < 2:
        return {}
    vals = {s: t.net_imbalance_dollars for s, t in latest.items()}
    arr = pd.Series(vals)
    mu, sd = arr.mean(), arr.std() + 1e-12
    return ((arr - mu) / sd).to_dict()


def top_movers(ticks: Iterable[ImbalanceTick], n: int = 5) -> dict:
    """{'buy': [(symbol, net_$), ...], 'sell': [...]} -- top-n largest net
    buy / sell imbalances across the latest reading per symbol."""
    latest = latest_per_symbol(ticks)
    ranked = sorted(latest.values(), key=lambda t: t.net_imbalance_dollars, reverse=True)
    buys = [(t.symbol, t.net_imbalance_dollars) for t in ranked[:n] if t.net_imbalance_dollars > 0]
    sells = [(t.symbol, t.net_imbalance_dollars) for t in ranked[-n:] if t.net_imbalance_dollars < 0]
    sells = sorted(sells, key=lambda kv: kv[1])
    return {"buy": buys, "sell": sells}


# --------------------------------------------------------------------------
# Bloomberg DAPI — market price feed
# --------------------------------------------------------------------------
def _ensure_pandas(obj):
    """Some `xbbg`/environment combinations have been observed returning a
    polars DataFrame from `blp.bdib`/`blp.bdh` instead of pandas (e.g. when
    polars is installed alongside xbbg -- this project's requirements.txt
    installs both) -- a polars `DataFrame` has no `.empty` attribute, which
    surfaces as a confusing `AttributeError` deep in normalization code
    that otherwise assumes pandas. Convert defensively via `.to_pandas()`
    (present on polars DataFrames) before doing anything else; a real
    pandas DataFrame (or None) passes through unchanged."""
    if obj is None or isinstance(obj, pd.DataFrame):
        return obj
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return obj


def _normalize_time_to_eastern(times: pd.Series, reference_now: "datetime | None" = None) -> pd.Series:
    """Coerce Bloomberg intraday-bar timestamps to tz-naive US/Eastern
    wall-clock. `blp.bdib` was originally assumed to already return local
    exchange time -- confirmed wrong against a live Terminal on
    2026-07-24 (it returned tz-aware/UTC-ish timestamps).

    First attempt at a fix used a median-hour-of-day heuristic (assume
    already-Eastern if the naive hour looked like typical trading hours,
    else assume UTC). That heuristic is UNSOUND and was confirmed wrong
    again the same day: UTC trading hours (13:30-21:00) and Eastern
    trading hours (9:30-17:00) overlap for most of the session, so a bar
    that's actually UTC (e.g. hour 15, i.e. 15:00 UTC) still looks like a
    plausible Eastern hour and the heuristic left it unconverted --
    reproducing the exact "no bar at/before {now} yet" symptom it was
    meant to fix.

    This version disambiguates using the one fact we actually know for
    certain: the true current Eastern moment (`reference_now`, defaults
    to `datetime.now(EST)` if not supplied). For tz-aware input, convert
    directly -- no ambiguity. For naive input, try both interpretations
    of the LATEST timestamp (as-is vs. as UTC-then-converted) and pick
    whichever lands at-or-before `reference_now` and closest to it --
    that is what "the most recent live bar" has to mean, so it is a much
    stronger signal than a static hour-of-day guess."""
    t = pd.to_datetime(times)
    if t.dt.tz is not None:
        return t.dt.tz_convert("America/New_York").dt.tz_localize(None)
    if len(t) == 0:
        return t
    if reference_now is None:
        reference_now = datetime.now(EST).replace(tzinfo=None)

    last_raw = t.iloc[-1]
    last_as_utc = pd.Timestamp(last_raw, tz="UTC").tz_convert("America/New_York").tz_localize(None)

    def _score(ts) -> tuple:
        delta = (reference_now - ts).total_seconds()
        return (0 if delta >= 0 else 1, abs(delta))  # prefer not-in-the-future, then closest to now

    if _score(last_as_utc) < _score(last_raw):
        return t.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)
    return t


class BloombergBarFeed:
    """Lightweight wrapper around the Bloomberg Desktop API (`xbbg`) for
    intraday 1-minute bars / live ticks on targeted tickers.

    Polls `xbbg.blp.bdib` (intraday bar request) at `poll_seconds` and
    yields only newly-completed bars per ticker as `PriceBar`s. Requires a
    live, logged-in Bloomberg Terminal session (Desktop API, localhost:8194)
    — see `fetch_spy_bloomberg.py` for the same constraint and setup notes.
    """

    def __init__(
        self,
        tickers: Iterable[str] = tuple(DEFAULT_ETFS),
        freq_minutes: int = 1,
        poll_seconds: float = 15.0,
        bbg_suffix: str = "US Equity",
    ):
        self.tickers = list(tickers)
        self.freq_minutes = freq_minutes
        self.poll_seconds = poll_seconds
        self.bbg_suffix = bbg_suffix
        self._last_ts: dict[str, datetime] = {}

    @staticmethod
    def parse_bar_row(symbol: str, row: pd.Series, ts: datetime) -> PriceBar:
        """Parse one normalized `blp.bdib` row (columns: open, high, low,
        close, volume) into a `PriceBar`."""
        return PriceBar(
            timestamp=ts, symbol=symbol,
            open=float(row["open"]), high=float(row["high"]),
            low=float(row["low"]), close=float(row["close"]),
            volume=float(row.get("volume", 0.0)), source="bloomberg",
        )

    @staticmethod
    def normalize_bdib_frame(raw, ticker: str, reference_now: "datetime | None" = None) -> pd.DataFrame:
        """`xbbg.blp.bdib` returns a DatetimeIndex frame whose columns are
        either flat (open/high/low/close/volume) for a single ticker, or a
        (ticker, field) MultiIndex for multi-ticker calls -- normalize both
        shapes to a flat DataFrame with a `time` column + those five
        lowercase field columns, `time` coerced to tz-naive US/Eastern
        wall-clock (see `_normalize_time_to_eastern` -- confirmed live
        2026-07-24 that `blp.bdib` on this setup returns naive UTC-ish
        timestamps, NOT already-local Eastern time as originally assumed;
        `compute_live_features` was rejecting every bar as "in the future"
        relative to `now`). `raw` is converted from polars to pandas first
        if needed (see `_ensure_pandas`)."""
        df = _ensure_pandas(raw)
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            level0 = out.columns.get_level_values(0)
            match = [c for c in level0.unique() if ticker.split()[0].upper() in str(c).upper()]
            key = match[0] if match else level0.unique()[0]
            out = out[key]
        out.columns = [str(c).lower() for c in out.columns]
        if "time" not in out.columns:
            if isinstance(out.index, pd.RangeIndex):
                raise ValueError("normalize_bdib_frame: no 'time' column and no usable "
                                 "DatetimeIndex in the Bloomberg response -- inspect `raw` "
                                 "and adjust this normalization (schema may have changed).")
            out = out.reset_index()
            out = out.rename(columns={out.columns[0]: "time"})  # former DatetimeIndex, whatever it was named
        out["time"] = _normalize_time_to_eastern(out["time"], reference_now=reference_now)
        return out[["time", "open", "high", "low", "close", "volume"]]

    def _poll_ticker(self, ticker: str) -> list[PriceBar]:
        _require_bloomberg_terminal()
        from xbbg import blp  # local import: only needed with a live Terminal session

        today = eastern_today().isoformat()
        raw = blp.bdib(
            ticker=f"{ticker} {self.bbg_suffix}", dt=today,
            session="allday", interval=self.freq_minutes,
        )
        df = self.normalize_bdib_frame(raw, ticker)
        if df.empty:
            return []
        bars = []
        last_seen = self._last_ts.get(ticker)
        for _, row in df.iterrows():
            ts = row["time"].to_pydatetime() if hasattr(row["time"], "to_pydatetime") else row["time"]
            if last_seen is not None and ts <= last_seen:
                continue
            bars.append(self.parse_bar_row(ticker, row, ts))
        if bars:
            self._last_ts[ticker] = bars[-1].timestamp
        return bars

    def fetch_today_bars(self, ticker: Optional[str] = None) -> pd.DataFrame:
        """Today's 1-minute bars so far for one ticker (default: the first
        configured ticker), as a flat DataFrame with columns
        date/time/open/high/low/close/volume -- same contract as
        `MassiveSpyBarFeed.fetch_today_bars`, so `live_stage1_feed.py` can
        use either source interchangeably."""
        _require_bloomberg_terminal()
        from xbbg import blp  # local import: only needed with a live Terminal session

        ticker = ticker or self.tickers[0]
        today = eastern_today().isoformat()
        raw = blp.bdib(ticker=f"{ticker} {self.bbg_suffix}", dt=today,
                       session="allday", interval=self.freq_minutes)
        df = self.normalize_bdib_frame(raw, ticker)
        if not df.empty:
            df["date"] = pd.to_datetime(df["time"]).dt.normalize()
        return df

    async def stream(self) -> AsyncIterator[PriceBar]:
        while True:
            for ticker in self.tickers:
                try:
                    bars = await asyncio.to_thread(self._poll_ticker, ticker)
                except Exception as exc:
                    print(f"[BloombergBarFeed] {ticker} poll failed: {exc}")
                    bars = []
                for bar in bars:
                    yield bar
            await asyncio.sleep(self.poll_seconds)


# --------------------------------------------------------------------------
# Alpaca (free IEX feed) — 1-minute bars, Bloomberg-free fallback
# --------------------------------------------------------------------------
class AlpacaBarFeed:
    """1-minute SPY (or other ticker) bars via Alpaca's free IEX data feed
    (`alpaca-py`). No Bloomberg Terminal required -- just a free Alpaca
    account (paper or live trading, doesn't matter for market data) and an
    API key pair.

    Same contract as `BloombergBarFeed`/`MassiveSpyBarFeed`:
    `fetch_today_bars()` re-fetches the day's bars-so-far in one call each
    time it's polled (simpler and self-healing, matching the other two
    feeds' convention -- a dropped poll just gets caught up on the next
    one) rather than a persistent websocket subscription. Alpaca's free
    tier *does* support real-time websocket streaming too, but the
    request/response polling shape is what `live_stage1_feed.py` and the
    Streamlit dashboard already expect from every price feed, so this
    class matches that rather than introducing a second, inconsistent
    pattern.

    IEX is a single exchange (~2.5% of consolidated US equity volume), not
    the full SIP consolidated tape Bloomberg/Massive draw from -- for a
    name as liquid as SPY this tracks the true tape closely, but is not
    byte-identical to a SIP-sourced bar. Treat as "close enough for a live
    read," not a source swapped in silently for anything that depends on
    exact historical reproducibility (that stays Bloomberg-sourced,
    per `fetch_spy_bloomberg.py`/`fetch_spy_massive.py`).
    """

    def __init__(
        self,
        tickers: Iterable[str] = tuple(DEFAULT_ETFS),
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        feed: str = "iex",
        poll_seconds: float = 15.0,
    ):
        self.tickers = list(tickers)
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        self.feed = feed
        self.poll_seconds = poll_seconds
        self._last_ts: dict[str, datetime] = {}

    @staticmethod
    def bars_to_dataframe(bars: Iterable) -> pd.DataFrame:
        """Pure conversion: `alpaca-py` `Bar` objects (or dict-likes with
        timestamp/open/high/low/close/volume) -> a DataFrame with columns
        date/time/open/high/low/close/volume, tz-naive US/Eastern
        wall-clock -- same contract as `MassiveSpyBarFeed.aggs_to_dataframe`.
        Split out so it's unit-testable without a live Alpaca connection."""
        rows = []
        for b in bars:
            get = (lambda k: b.get(k)) if isinstance(b, dict) else (lambda k: getattr(b, k))
            ts = pd.Timestamp(get("timestamp"))
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            ts = ts.tz_convert(EST).tz_localize(None)
            rows.append({"time": ts, "open": get("open"), "high": get("high"), "low": get("low"),
                        "close": get("close"), "volume": get("volume")})
        if not rows:
            return pd.DataFrame(columns=["date", "time", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
        df["date"] = df["time"].dt.normalize()
        return df

    def _client(self):
        if not self.api_key or not self.secret_key:
            raise RuntimeError(
                "ALPACA_API_KEY/ALPACA_SECRET_KEY not set -- see .env.example "
                "(sign up free at https://alpaca.markets, no funding required "
                "for market data access)."
            )
        from alpaca.data.historical.stock import StockHistoricalDataClient  # local import: only needed with alpaca-py installed

        return StockHistoricalDataClient(self.api_key, self.secret_key)

    def fetch_today_bars(self, ticker: Optional[str] = None) -> pd.DataFrame:
        """Today's 1-minute bars so far for one ticker (default: the first
        configured ticker), as a flat DataFrame with columns
        date/time/open/high/low/close/volume -- same contract as
        `BloombergBarFeed.fetch_today_bars`/`MassiveSpyBarFeed.fetch_today_bars`,
        so `live_stage1_feed.py` can use any of the three interchangeably."""
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        ticker = ticker or self.tickers[0]
        today = eastern_today()
        start_utc = EST.localize(datetime.combine(today, time(0, 0))).astimezone(pytz.utc)
        client = self._client()
        req = StockBarsRequest(
            symbol_or_symbols=ticker, timeframe=TimeFrame.Minute,
            start=start_utc, feed=DataFeed(self.feed),
        )
        barset = client.get_stock_bars(req)
        bars = barset.data.get(ticker, [])
        return self.bars_to_dataframe(bars)

    async def stream(self) -> AsyncIterator[PriceBar]:
        while True:
            for ticker in self.tickers:
                try:
                    df = await asyncio.to_thread(self.fetch_today_bars, ticker)
                except Exception as exc:
                    print(f"[AlpacaBarFeed] {ticker} poll failed: {exc}")
                    df = pd.DataFrame()
                last_seen = self._last_ts.get(ticker)
                for _, row in df.iterrows():
                    ts = row["time"]
                    if last_seen is not None and ts <= last_seen:
                        continue
                    yield PriceBar(timestamp=ts, symbol=ticker, open=row["open"], high=row["high"],
                                  low=row["low"], close=row["close"], volume=row["volume"], source="alpaca")
                if not df.empty:
                    self._last_ts[ticker] = df["time"].iloc[-1]
            await asyncio.sleep(self.poll_seconds)


# --------------------------------------------------------------------------
# Bloomberg DAPI — lagged VIX close (Stage 1 feature input)
# --------------------------------------------------------------------------
def normalize_bdh_frame(raw) -> pd.DataFrame:
    """`xbbg.blp.bdh` (historical daily bars/fields) returns a DatetimeIndex
    frame whose columns are either flat (a single field name, OR -- seen
    live 2026-07-24 for a single-ticker/single-field call -- flat but
    labeled by TICKER, e.g. "VIX Index", not by field name) or a (ticker,
    field) MultiIndex -- normalize all of these to a flat DataFrame with
    `date`/`px_last` columns. `raw` is converted from polars to pandas
    first if needed (see `_ensure_pandas`).

    px_last column selection prefers an exact known field name, then
    falls back to whichever remaining column actually holds numeric data
    -- NOT just "the first remaining column" -- because that naive
    fallback picked a non-numeric ticker-label column live (crashed with
    "could not convert string to float: 'VIX Index'": the column ended up
    named/selected by ticker, not field, for this single-ticker query
    shape)."""
    df = _ensure_pandas(raw)
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date", "px_last"])
    out = df.copy()
    out.columns = [str(c[-1]).lower() if isinstance(c, tuple) else str(c).lower() for c in out.columns]
    if "date" not in out.columns:
        if isinstance(out.index, pd.RangeIndex):
            raise ValueError("normalize_bdh_frame: no 'date' column and no usable DatetimeIndex "
                             "in the Bloomberg response -- inspect `raw` and adjust this "
                             "normalization (schema may have changed).")
        out = out.reset_index()
        out = out.rename(columns={out.columns[0]: "date"})
    if "px_last" not in out.columns:
        remaining = [c for c in out.columns if c != "date"]
        if not remaining:
            raise ValueError("normalize_bdh_frame: no price field column found in the "
                             "Bloomberg response.")
        preferred = [c for c in remaining if c in ("px_last", "close", "last_price", "value")]
        numeric = [c for c in remaining if pd.to_numeric(out[c], errors="coerce").notna().any()]
        pick = preferred[0] if preferred else (numeric[0] if numeric else remaining[0])
        out = out.rename(columns={pick: "px_last"})
    out = out[["date", "px_last"]].copy()
    out["px_last"] = pd.to_numeric(out["px_last"], errors="coerce")
    if out["px_last"].isna().all():
        raise ValueError(
            f"normalize_bdh_frame: 'px_last' has no numeric values after selection -- "
            f"original Bloomberg response columns were {list(df.columns)}; schema may "
            "have changed, inspect `raw` and adjust this normalization."
        )
    out["date"] = pd.to_datetime(out["date"])
    if out["date"].dt.tz is not None:
        # Daily bars: a tz-aware midnight timestamp still labels a single
        # calendar day -- strip the tz rather than convert it, which would
        # shift the date across the day boundary for no reason.
        out["date"] = out["date"].dt.tz_localize(None)
    return out


def most_recent_close_before(df: pd.DataFrame, as_of: date) -> float:
    """The last `px_last` in `df` (columns: date, px_last) strictly before
    `as_of` -- the "lagged" (most recently COMPLETED session's) close."""
    sub = df[pd.to_datetime(df["date"]).dt.date < as_of]
    if sub.empty:
        raise RuntimeError(f"no close found strictly before {as_of}")
    return float(sub.sort_values("date")["px_last"].iloc[-1])


def fetch_bloomberg_vix_close_lagged(as_of: Optional[date] = None, lookback_days: int = 10) -> float:
    """The lagged VIX close (`VIX Index`), sourced live from Bloomberg
    (`xbbg.blp.bdh`) rather than a static file. `as_of` defaults to today;
    returns the most recent COMPLETED session's close strictly before it,
    matching `build_feature_table.load_vix`'s `.shift(1)` semantics."""
    _require_bloomberg_terminal()
    from xbbg import blp  # local import: only needed with a live Terminal session

    as_of = as_of or eastern_today()
    start = as_of - timedelta(days=lookback_days)
    raw = blp.bdh(tickers="VIX Index", flds="PX_LAST",
                 start_date=start.isoformat(), end_date=as_of.isoformat())
    df = normalize_bdh_frame(raw)
    return most_recent_close_before(df, as_of)


# --------------------------------------------------------------------------
# Massive API — live SPY market data (Stage 1 feature inputs)
# --------------------------------------------------------------------------
class MassiveSpyBarFeed:
    """Live SPY 1-minute bars via Massive's custom-bars endpoint
    (`RESTClient.get_aggs`) -- the simplest endpoint this project's API key
    is entitled to (`get_last_trade` returned NOT_AUTHORIZED during
    integration; `get_aggs` and `get_snapshot_ticker` both work).

    Each poll re-fetches the *entire trading day's* 1-minute bars in one
    call rather than tracking incremental state -- simpler and self-healing
    (a dropped poll just gets caught up on the next one), and it's exactly
    the shape `q005_technical_state_features._per_day_technical_features`
    expects (a full day of bars, session-cumulative features recomputed
    from the start each time). See `src/live_stage1_feed.py` for that reuse.
    """

    def __init__(self, ticker: str = "SPY", poll_seconds: float = 15.0, api_key: Optional[str] = None):
        self.ticker = ticker
        self.poll_seconds = poll_seconds
        self.api_key = api_key or os.environ.get("MASSIVE_API_KEY")
        self._last_ts: Optional[datetime] = None

    @staticmethod
    def aggs_to_dataframe(aggs: Iterable) -> pd.DataFrame:
        """Pure conversion: Massive `Agg` objects (or any object/dict with
        timestamp[ms]/open/high/low/close/volume) -> a DataFrame with
        columns date/time/open/high/low/close/volume (tz-naive US/Eastern
        wall-clock times, matching this project's historical convention).
        Split out from `fetch_today_bars` so it's unit-testable without a
        live Massive connection."""
        rows = []
        for a in aggs:
            get = (lambda k: a.get(k)) if isinstance(a, dict) else (lambda k: getattr(a, k))
            ts = datetime.fromtimestamp(get("timestamp") / 1000.0, tz=pytz.utc).astimezone(EST).replace(tzinfo=None)
            rows.append({"time": ts, "open": get("open"), "high": get("high"), "low": get("low"),
                        "close": get("close"), "volume": get("volume")})
        if not rows:
            return pd.DataFrame(columns=["date", "time", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
        df["date"] = df["time"].dt.normalize()
        return df

    def fetch_today_bars(self) -> pd.DataFrame:
        """Today's 1-minute bars so far, fetched fresh from Massive (one
        `get_aggs` call for the whole day, then `aggs_to_dataframe`)."""
        from massive import RESTClient

        client = RESTClient(self.api_key)
        today = eastern_today().isoformat()
        aggs = list(client.get_aggs(self.ticker, 1, "minute", today, today, limit=1000))
        return self.aggs_to_dataframe(aggs)

    async def stream(self) -> AsyncIterator[PriceBar]:
        while True:
            try:
                df = await asyncio.to_thread(self.fetch_today_bars)
            except Exception as exc:
                print(f"[MassiveSpyBarFeed] poll failed: {exc}")
                df = pd.DataFrame()
            for _, row in df.iterrows():
                ts = row["time"]
                if self._last_ts is not None and ts <= self._last_ts:
                    continue
                yield PriceBar(timestamp=ts, symbol=self.ticker, open=row["open"], high=row["high"],
                              low=row["low"], close=row["close"], volume=row["volume"], source="massive")
            if not df.empty:
                self._last_ts = df["time"].iloc[-1]
            await asyncio.sleep(self.poll_seconds)


# --------------------------------------------------------------------------
# Replay fallback — dry-run streamer over historical CSVs
# --------------------------------------------------------------------------
class ReplayFeed:
    """Offline dry-run streamer: replays a historical session's imbalance
    ticks and price bars in chronological order, tick-by-tick, honoring the
    original inter-event timing (scaled by `speed`) so downstream code sees
    the same cadence a live session would produce.

    `speed`: wall-clock seconds per data-second (0 = as fast as possible,
    1.0 = real time, 60.0 = 60x realtime replay).
    """

    def __init__(self, imbalance_df: pd.DataFrame, price_df: pd.DataFrame, speed: float = 0.0):
        self._events = self._merge_events(imbalance_df, price_df)
        self.speed = speed

    @staticmethod
    def _merge_events(imbalance_df: pd.DataFrame, price_df: pd.DataFrame) -> list:
        events: list[tuple[datetime, object]] = []
        if imbalance_df is not None and not imbalance_df.empty:
            for _, r in imbalance_df.iterrows():
                ts = pd.to_datetime(r["SaleTime"]).to_pydatetime()
                side = r["Side"] if "Side" in r else (1 if r.get("dollar_imbalance", 0) >= 0 else -1)
                ref_price = float(r.get("LastSale", 0.0))
                net = float(r["dollar_imbalance"]) if "dollar_imbalance" in r else float(r.get("Imbalance", 0.0)) * side * ref_price
                tick = ImbalanceTick(timestamp=ts, symbol=str(r["Symbol"]), net_imbalance_dollars=net,
                                     side=int(side), reference_price=ref_price, source="replay")
                events.append((ts, tick))
        if price_df is not None and not price_df.empty:
            for _, r in price_df.iterrows():
                ts = pd.to_datetime(r["time"]).to_pydatetime()
                bar = PriceBar(timestamp=ts, symbol=str(r.get("symbol", r.get("ticker", "SPY"))),
                                open=float(r["open"]), high=float(r["high"]), low=float(r["low"]),
                                close=float(r["close"]), volume=float(r.get("volume", 0.0)), source="replay")
                events.append((ts, bar))
        events.sort(key=lambda e: e[0])
        return events

    @classmethod
    def from_csv_dir(cls, imbalance_dir: str | Path, price_csv: str | Path, session_date: str,
                      speed: float = 0.0) -> "ReplayFeed":
        """Build a `ReplayFeed` for one historical session date from a raw
        imbalance directory (schema: `load_imbalances.py`) + a single price
        CSV (columns: time, open, high, low, close, volume)."""
        import load_imbalances as li  # sibling package on sys.path (src/)

        imb = li.load_raw_imbalance_dir(imbalance_dir)
        imb = imb[imb["SaleTime"].dt.date.astype(str) == session_date]
        prices = pd.read_csv(price_csv)
        prices["time"] = pd.to_datetime(prices["time"])
        prices = prices[prices["time"].dt.date.astype(str) == session_date]
        return cls(imb, prices, speed=speed)

    async def stream(self) -> AsyncIterator[ImbalanceTick | PriceBar]:
        prev_ts = None
        for ts, event in self._events:
            if self.speed > 0 and prev_ts is not None:
                gap = (ts - prev_ts).total_seconds() * self.speed
                if gap > 0:
                    await asyncio.sleep(gap)
            prev_ts = ts
            yield event


# --------------------------------------------------------------------------
# fan-in helper — merge N async feeds onto one queue without blocking
# --------------------------------------------------------------------------
async def merge_into_queue(feeds: Iterable, queue: "asyncio.Queue", stop_event: Optional[asyncio.Event] = None) -> None:
    """Run each feed's `stream()` as its own asyncio task, pushing every
    yielded record onto a shared queue. A slow/blocked feed (e.g. a stalled
    REST poll) only delays its own task, never the others or the consumer
    draining the queue — that's the whole point of keeping this async
    rather than a single blocking loop."""

    async def _pump(feed):
        async for record in feed.stream():
            await queue.put(record)
            if stop_event is not None and stop_event.is_set():
                return

    tasks = [asyncio.create_task(_pump(f)) for f in feeds]
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for t in tasks:
            t.cancel()
        raise
