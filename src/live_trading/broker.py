"""Thin wrapper around Alpaca's Trading API (`alpaca-py`,
`alpaca.trading.client.TradingClient`) for order placement, position
tracking, and flattening -- the one place in this repo that actually talks
to a broker.

Safety posture: `paper: bool` is a REQUIRED constructor argument (no
silent default) -- every call site must say explicitly which account it's
hitting, so "which account am I trading in" is never ambiguous or
accidentally inherited from an env var. Alpaca additionally enforces
paper/live key separation server-side (a paper-tier API key physically
cannot authenticate against the live endpoint and vice versa), so going
from paper to real money requires BOTH swapping in a different key pair
AND deliberately passing `paper=False` -- never a single switch.

Every order-affecting call logs intent *before* the API call and the
result *after* (`audit_log.log_event`), so a crash mid-call still leaves a
record of what was attempted, not just what succeeded.
"""

from __future__ import annotations

from typing import Optional

from live_trading import audit_log

DEFAULT_TIME_IN_FORCE = "day"


class AlpacaBroker:
    def __init__(self, api_key: str, secret_key: str, paper: bool):
        from alpaca.trading.client import TradingClient

        self.paper = paper
        self._client = TradingClient(api_key, secret_key, paper=paper)

    # ---- read-only ----
    def account(self) -> dict:
        acct = self._client.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "status": str(acct.status),
        }

    def positions(self) -> list[dict]:
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": "long" if float(p.qty) > 0 else "short",
                "avg_entry_price": float(p.avg_entry_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
            }
            for p in self._client.get_all_positions()
        ]

    # ---- order placement / flattening ----
    def submit_market_order(self, symbol: str, qty: float, side: str,
                            time_in_force: str = DEFAULT_TIME_IN_FORCE) -> dict:
        """`side`: "buy" or "sell". `qty` must be > 0 -- direction comes
        from `side`, not from a signed quantity, to match Alpaca's own API
        shape and avoid a sign-convention mismatch bug class."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        if qty <= 0:
            raise ValueError(f"qty must be > 0, got {qty!r} -- direction comes from `side`")
        audit_log.log_event("order_intent", broker="alpaca", paper=self.paper,
                            symbol=symbol, qty=qty, side=side, order_type="market")
        req = MarketOrderRequest(
            symbol=symbol, qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce(time_in_force),
        )
        try:
            order = self._client.submit_order(req)
        except Exception as exc:
            audit_log.log_event("order_result", broker="alpaca", paper=self.paper,
                                symbol=symbol, qty=qty, side=side, status="error", error=str(exc))
            raise
        result = {"id": str(order.id), "symbol": order.symbol, "qty": float(order.qty),
                  "side": str(order.side), "status": str(order.status)}
        audit_log.log_event("order_result", broker="alpaca", paper=self.paper, **result)
        return result

    def close_position(self, symbol: str) -> Optional[dict]:
        audit_log.log_event("close_position_intent", broker="alpaca", paper=self.paper, symbol=symbol)
        try:
            order = self._client.close_position(symbol)
        except Exception as exc:
            audit_log.log_event("close_position_result", broker="alpaca", paper=self.paper,
                                symbol=symbol, status="error", error=str(exc))
            raise
        result = {"symbol": symbol, "order_id": str(getattr(order, "id", ""))}
        audit_log.log_event("close_position_result", broker="alpaca", paper=self.paper,
                            status="ok", **result)
        return result

    def close_all(self) -> list[dict]:
        """Flattens every open position AND cancels every open order --
        the kill switch's core primitive. Best-effort per position: one
        failure doesn't stop the rest from being attempted."""
        audit_log.log_event("close_all_intent", broker="alpaca", paper=self.paper)
        try:
            responses = self._client.close_all_positions(cancel_orders=True)
        except Exception as exc:
            audit_log.log_event("close_all_result", broker="alpaca", paper=self.paper,
                                status="error", error=str(exc))
            raise
        results = [{"symbol": getattr(r, "symbol", None), "status": getattr(r, "status", None)}
                  for r in responses]
        audit_log.log_event("close_all_result", broker="alpaca", paper=self.paper,
                            status="ok", closed=results)
        return results
