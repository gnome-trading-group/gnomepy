from __future__ import annotations

from dataclasses import dataclass, field

import jpype

from gnomepy.java.enums import Side, OrderType


@dataclass
class Intent:
    """Flat union intent — supports quoting, taking, or both simultaneously.

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

    exchange_id: int
    security_id: int
    bid_price: int = 0
    bid_size: int = 0
    ask_price: int = 0
    ask_size: int = 0
    take_side: Side | None = None
    take_size: int = 0
    take_order_type: OrderType | None = None
    take_limit_price: int = 0


@dataclass
class PositionInfo:
    """Python view of an OMS position."""

    exchange_id: int
    security_id: int
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


@dataclass
class RiskConfig:
    """Configuration for OMS risk policies."""

    max_notional_value: int | None = None


class OmsView:
    """Read-only Python view of the Java OMS state.

    Strategies use this to query positions and open orders
    during on_market_data() or on_execution_report() callbacks.
    """

    def __init__(self, java_oms):
        self._java = java_oms

    def get_position(self, exchange_id: int, security_id: int) -> PositionInfo | None:
        pos = self._java.getPosition(int(exchange_id), jpype.JLong(security_id))
        if pos is None:
            return None
        return _position_from_java(pos)

    def get_all_positions(self) -> list[PositionInfo]:
        result = []
        self._java.forEachPosition(lambda p: result.append(_position_from_java(p)))
        return result

    def get_order(self, client_oid: str) -> TrackedOrderInfo | None:
        tracked = self._java.getOrder(str(client_oid))
        if tracked is None:
            return None
        return _tracked_order_from_java(tracked)

    def get_open_orders(self) -> list[TrackedOrderInfo]:
        result = []
        self._java.forEachOpenOrder(lambda o: result.append(_tracked_order_from_java(o)))
        return result

    def get_open_orders_for(self, exchange_id: int, security_id: int) -> list[TrackedOrderInfo]:
        result = []
        self._java.forEachOpenOrderFor(
            int(exchange_id), jpype.JLong(security_id),
            lambda o: result.append(_tracked_order_from_java(o)),
        )
        return result


def _position_from_java(pos) -> PositionInfo:
    return PositionInfo(
        exchange_id=int(pos.getExchangeId()),
        security_id=int(pos.getSecurityId()),
        net_quantity=int(pos.getNetQuantity()),
        avg_entry_price=int(pos.getAvgEntryPrice()),
        realized_pnl=float(pos.getRealizedPnl()),
        total_fees=float(pos.getTotalFees()),
    )


def _tracked_order_from_java(tracked) -> TrackedOrderInfo:
    return TrackedOrderInfo(
        client_oid=str(tracked.getClientOid()),
        exchange_id=int(tracked.getExchangeId()),
        security_id=int(tracked.getSecurityId()),
        side=Side.from_java(tracked.getSide()),
        price=int(tracked.getOriginalOrder().price()),
        size=int(tracked.getOriginalOrder().size()),
        state=str(tracked.getState().name()),
        cumulative_qty=int(tracked.getCumulativeQty()),
        leaves_qty=int(tracked.getLeavesQty()),
        avg_fill_price=int(tracked.getAvgFillPrice()),
    )


def _build_java_oms(risk_config: RiskConfig, oid_generator=None):
    """Build a Java OrderManagementSystem from a Python RiskConfig.

    Args:
        risk_config: Risk policy configuration.
        oid_generator: Callable returning unique client OID strings.
            If None, uses a counter-based default.
    """
    RiskEngine = jpype.JClass("group.gnometrading.oms.risk.RiskEngine")
    RiskPolicyClass = jpype.JClass("group.gnometrading.oms.risk.RiskPolicy")
    OMS = jpype.JClass("group.gnometrading.oms.OrderManagementSystem")

    policy_list = []

    if risk_config.max_notional_value is not None:
        MaxNotionalValuePolicy = jpype.JClass("group.gnometrading.oms.risk.MaxNotionalValuePolicy")
        policy_list.append(MaxNotionalValuePolicy(jpype.JLong(risk_config.max_notional_value)))

    java_array = jpype.JArray(RiskPolicyClass)(len(policy_list))
    for i, p in enumerate(policy_list):
        java_array[i] = p

    engine = RiskEngine(java_array)

    if oid_generator is None:
        _counter = [0]
        def oid_generator():
            _counter[0] += 1
            return f"oms-{_counter[0]}"

    return OMS(engine, oid_generator)

