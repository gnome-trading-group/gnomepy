from __future__ import annotations

from dataclasses import dataclass

import jpype

from gnomepy.java.enums import Side, OrderType


class Intent:
    """Python wrapper around the Java SBE Intent message.

    Supports quoting, taking, or both simultaneously.

    Quote fields (passive resting orders):
        bid_price/bid_size: desired bid. size=0 means cancel existing bid.
        ask_price/ask_size: desired ask. size=0 means cancel existing ask.

    Take fields (aggressive IOC order):
        take_side/take_size: aggressive order. size=0 means no take this tick.
        take_order_type: MARKET or LIMIT (for IOC limit).
        take_limit_price: price for IOC limit takes.

    Usage:
        # Two-sided quote
        Intent(1, 100, bid_price=9950, bid_size=10, ask_price=10050, ask_size=10)

        # Quote + aggressive take
        Intent(1, 100, bid_price=9950, bid_size=10, ask_price=10050, ask_size=10,
               take_side=Side.BID, take_size=50, take_order_type=OrderType.MARKET)

        # Cancel all quotes
        Intent(1, 100)
    """

    _java_class = None

    @classmethod
    def _ensure_class(cls):
        if cls._java_class is None:
            cls._java_class = jpype.JClass("group.gnometrading.schemas.Intent")

    def __init__(
        self,
        exchange_id: int,
        security_id: int,
        strategy_id: int = 0,
        *,
        bid_price: int = 0,
        bid_size: int = 0,
        ask_price: int = 0,
        ask_size: int = 0,
        take_side: Side | None = None,
        take_size: int = 0,
        take_order_type: OrderType | None = None,
        take_limit_price: int = 0,
    ):
        self._ensure_class()
        self._java = self._java_class()
        enc = self._java.encoder
        enc.exchangeId(jpype.JInt(exchange_id))
        enc.securityId(jpype.JLong(security_id))
        enc.strategyId(jpype.JInt(strategy_id))
        enc.bidPrice(jpype.JLong(bid_price))
        enc.bidSize(jpype.JLong(bid_size))
        enc.askPrice(jpype.JLong(ask_price))
        enc.askSize(jpype.JLong(ask_size))
        if take_size > 0 and take_side is not None:
            enc.takeSide(take_side.to_java())
            enc.takeSize(jpype.JLong(take_size))
            enc.takeOrderType(take_order_type.to_java())
            enc.takeLimitPrice(jpype.JLong(take_limit_price))

    @property
    def raw(self):
        """The underlying Java SBE Intent object."""
        return self._java

    @property
    def exchange_id(self) -> int:
        return int(self._java.decoder.exchangeId())

    @property
    def security_id(self) -> int:
        return int(self._java.decoder.securityId())

    @property
    def strategy_id(self) -> int:
        return int(self._java.decoder.strategyId())

    @property
    def bid_price(self) -> int:
        return int(self._java.decoder.bidPrice())

    @property
    def bid_size(self) -> int:
        return int(self._java.decoder.bidSize())

    @property
    def ask_price(self) -> int:
        return int(self._java.decoder.askPrice())

    @property
    def ask_size(self) -> int:
        return int(self._java.decoder.askSize())

    @property
    def take_side(self) -> Side | None:
        java_side = self._java.decoder.takeSide()
        s = Side.from_java(java_side)
        return None if s == Side.NONE else s

    @property
    def take_size(self) -> int:
        return int(self._java.decoder.takeSize())

    @property
    def take_order_type(self) -> OrderType | None:
        java_ot = self._java.decoder.takeOrderType()
        name = str(java_ot.name())
        if name == "NULL_VAL":
            return None
        return OrderType(name)

    @property
    def take_limit_price(self) -> int:
        return int(self._java.decoder.takeLimitPrice())


@dataclass
class PositionInfo:
    """Python view of an OMS position."""

    listing_id: int
    net_quantity: int
    avg_entry_price: int
    realized_pnl: float
    total_fees: float


@dataclass
class TrackedOrderInfo:
    """Python view of a tracked order."""

    client_oid: str
    exchange_id: int
    security_id: int
    side: Side
    price: int
    size: int
    state: str
    cumulative_qty: int
    leaves_qty: int
    avg_fill_price: int


class OmsView:
    """Read-only Python view of the Java OMS state.

    Strategies use this to query positions and open orders
    during on_market_data() or on_execution_report() callbacks.
    """

    def __init__(self, java_oms, security_master, strategy_id: int = 0):
        self._java = java_oms
        self._security_master = security_master
        self._strategy_id = int(strategy_id)

    def _resolve_listing_id(self, exchange_id: int, security_id: int) -> int:
        listing = self._security_master.getListing(int(exchange_id), int(security_id))
        return int(listing.listingId())

    def get_position(self, exchange_id: int, security_id: int) -> PositionInfo | None:
        listing_id = self._resolve_listing_id(exchange_id, security_id)
        pos = self._java.getPosition(int(listing_id))
        if pos is None:
            return None
        return _position_from_java(pos)

    def get_effective_quantity(
        self,
        exchange_id: int,
        security_id: int,
        strategy_id: int | None = None,
    ) -> int:
        """Net position + inflight orders (what position will be if all pending orders fill).

        `strategy_id` defaults to the id this OmsView was bound to by the runner —
        strategies should not need to pass it.
        """
        sid = self._strategy_id if strategy_id is None else int(strategy_id)
        listing_id = self._resolve_listing_id(exchange_id, security_id)
        return int(self._java.getEffectiveQuantity(sid, int(listing_id)))

    def get_all_positions(self) -> list[PositionInfo]:
        result = []
        self._java.getPositionTracker().forEachPosition(
            lambda p: result.append(_position_from_java(p))
        )
        return result

    def get_order(self, client_oid: str) -> TrackedOrderInfo | None:
        tracked = self._java.getOrder(jpype.JLong(int(client_oid)))
        if tracked is None:
            return None
        return _tracked_order_from_java(tracked)

    def get_open_orders(self) -> list[TrackedOrderInfo]:
        result = []

        def _collect(o):
            if not o.getState().isTerminal():
                result.append(_tracked_order_from_java(o))

        self._java.getOrderStateManager().forEachOrder(_collect)
        return result

    def get_open_orders_for(self, exchange_id: int, security_id: int) -> list[TrackedOrderInfo]:
        result = []

        def _collect(o):
            if (
                not o.getState().isTerminal()
                and int(o.getExchangeId()) == exchange_id
                and int(o.getSecurityId()) == security_id
            ):
                result.append(_tracked_order_from_java(o))

        self._java.getOrderStateManager().forEachOrder(_collect)
        return result


def _position_from_java(pos) -> PositionInfo:
    return PositionInfo(
        listing_id=int(pos.listingId),
        net_quantity=int(pos.netQuantity),
        avg_entry_price=int(pos.getAvgEntryPrice()),
        realized_pnl=float(pos.realizedPnl),
        total_fees=float(pos.totalFees),
    )


def _tracked_order_from_java(tracked) -> TrackedOrderInfo:
    return TrackedOrderInfo(
        client_oid=str(tracked.getClientOidCounter()),
        exchange_id=int(tracked.getExchangeId()),
        security_id=int(tracked.getSecurityId()),
        side=Side.from_java(tracked.getSide()),
        price=int(tracked.getPrice()),
        size=int(tracked.getSize()),
        state=str(tracked.getState().name()),
        cumulative_qty=int(tracked.getFilledQty()),
        leaves_qty=int(tracked.getLeavesQty()),
        avg_fill_price=int(tracked.getAvgFillPrice()),
    )
