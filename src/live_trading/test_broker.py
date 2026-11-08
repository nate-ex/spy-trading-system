"""Unit tests for AlpacaBroker -- no live network. Constructs a real
`TradingClient` (confirmed not to eagerly validate credentials/hit the
network at construction time) then swaps in a fake for the actual calls,
so `AlpacaBroker`'s own translation/audit-logging logic is exercised
without a real Alpaca connection."""

import pytest

from live_trading import audit_log
from live_trading.broker import AlpacaBroker


@pytest.fixture(autouse=True)
def _isolated_log_dir(tmp_path, monkeypatch):
    """Every test in this file goes through real `AlpacaBroker` order
    methods, which call `audit_log.log_event()` unconditionally -- without
    this, test runs would silently write fake order events into the
    repo's real, COMMITTED audit log. Isolate every test, not just the one
    that explicitly checks log contents."""
    monkeypatch.setattr(audit_log, "LOG_DIR", tmp_path)
    yield


class _FakeOrder:
    def __init__(self, id="order-1", symbol="SPY", qty=10, side="buy", status="accepted"):
        self.id, self.symbol, self.qty, self.side, self.status = id, symbol, qty, side, status


class _FakePosition:
    def __init__(self, symbol="SPY", qty=10, avg_entry_price=550.0, market_value=5500.0, unrealized_pl=12.5):
        self.symbol, self.qty = symbol, qty
        self.avg_entry_price, self.market_value, self.unrealized_pl = avg_entry_price, market_value, unrealized_pl


class _FakeAccount:
    equity, cash, buying_power, status = "10000.0", "5000.0", "20000.0", "ACTIVE"


class _FakeClosePositionResponse:
    def __init__(self, symbol="SPY", status=200):
        self.symbol, self.status = symbol, status


class _FakeTradingClient:
    def __init__(self):
        self.submitted = []
        self.closed_positions = []
        self.closed_all = False

    def get_account(self):
        return _FakeAccount()

    def get_all_positions(self):
        return [_FakePosition()]

    def submit_order(self, req):
        self.submitted.append(req)
        return _FakeOrder(symbol=req.symbol, qty=req.qty, side=str(req.side))

    def close_position(self, symbol, close_options=None):
        self.closed_positions.append(symbol)
        return _FakeOrder(symbol=symbol)

    def close_all_positions(self, cancel_orders=None):
        self.closed_all = True
        return [_FakeClosePositionResponse()]


def _broker() -> tuple[AlpacaBroker, _FakeTradingClient]:
    broker = AlpacaBroker("fake-key", "fake-secret", paper=True)
    fake = _FakeTradingClient()
    broker._client = fake
    return broker, fake


def test_account_translates_fields():
    broker, _ = _broker()
    acct = broker.account()
    assert acct == {"equity": 10000.0, "cash": 5000.0, "buying_power": 20000.0, "status": "ACTIVE"}


def test_positions_translates_fields():
    broker, _ = _broker()
    positions = broker.positions()
    assert positions == [{"symbol": "SPY", "qty": 10.0, "side": "long", "avg_entry_price": 550.0,
                          "market_value": 5500.0, "unrealized_pl": 12.5}]


def test_submit_market_order_rejects_nonpositive_qty():
    broker, _ = _broker()
    with pytest.raises(ValueError):
        broker.submit_market_order("SPY", 0, "buy")
    with pytest.raises(ValueError):
        broker.submit_market_order("SPY", -5, "buy")


def test_submit_market_order_submits_and_returns_result():
    broker, fake = _broker()
    result = broker.submit_market_order("SPY", 10, "buy")
    assert len(fake.submitted) == 1
    assert fake.submitted[0].symbol == "SPY"
    assert result["symbol"] == "SPY" and result["qty"] == 10.0


def test_close_position_calls_client():
    broker, fake = _broker()
    broker.close_position("SPY")
    assert fake.closed_positions == ["SPY"]


def test_close_all_calls_client_and_returns_results():
    broker, fake = _broker()
    results = broker.close_all()
    assert fake.closed_all is True
    assert results == [{"symbol": "SPY", "status": 200}]


def test_broker_logs_order_events(tmp_path):
    broker, _ = _broker()
    broker.submit_market_order("SPY", 5, "buy")
    log_files = list(tmp_path.glob("live_orders_*.jsonl"))
    assert len(log_files) == 1
    lines = log_files[0].read_text().strip().splitlines()
    assert len(lines) == 2  # order_intent + order_result
    assert '"order_intent"' in lines[0]
    assert '"order_result"' in lines[1]
