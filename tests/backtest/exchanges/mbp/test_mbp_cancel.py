import pytest
from gnomepy.backtest.exchanges.mbp.mbp import MBPSimulatedExchange
from gnomepy.backtest.fee import FeeModel
from gnomepy.backtest.latency import LatencyModel
from gnomepy.backtest.queues.base import QueueModel
from gnomepy.data.types import (
    Order, OrderType, TimeInForce, OrderStatus, ExecType,
    MBP10, MBP1, BidAskPair
)
from tests.backtest.exchanges.mbp.test_mbp_book import (
    DummyQueueModel, make_order, make_trade, make_bid_ask_pair
)


class DummyFeeModel(FeeModel):
    def calculate_fee(self, notional: int, is_maker: bool) -> int:
        return 0  # No fees for testing


class DummyLatencyModel(LatencyModel):
    def simulate(self) -> int:
        return 0  # No latency for testing


def setup_exchange_with_orders(orders=None, market_data=None):
    """Helper to set up an exchange with orders and market data"""
    exchange = MBPSimulatedExchange(
        fee_model=DummyFeeModel(),
        network_latency=DummyLatencyModel(),
        order_processing_latency=DummyLatencyModel(),
        queue_model=DummyQueueModel(),
    )
    
    # Add market data if provided
    if market_data:
        for data in market_data:
            exchange.on_market_data(data)
    
    # Add orders if provided
    if orders:
        for order in orders:
            exchange.submit_order(order)
    
    return exchange


def make_market_data(action, levels=None):
    """Helper to create market data for the exchange"""
    if levels is None:
        levels = []
    
    return MBP10(
        action=action,
        side="B",  # Not used for market updates
        price=0,   # Not used for market updates
        size=0,    # Not used for market updates
        timestamp_event=0,
        timestamp_recv=0,
        timestamp_sent=0,
        exchange_id=1,
        security_id=1,
        flags=[],
        sequence=0,
        depth=0,
        levels=levels,
    )


@pytest.mark.parametrize(
    "orders, market_data, client_oid_to_cancel, expected_exec_type, expected_order_status, expected_success",
    [
        # === BASIC CANCELLATION TESTS ===
        
        # Successful cancellation of a single order
        (
            [make_order(price=100, size=10, side="B", client_oid="ORDER_1")],
            [],
            "ORDER_1",
            ExecType.CANCELED,
            OrderStatus.CANCELED,
            True,
        ),
        
        # Successful cancellation of bid order
        (
            [make_order(price=100, size=5, side="B", client_oid="BID_1")],
            [],
            "BID_1",
            ExecType.CANCELED,
            OrderStatus.CANCELED,
            True,
        ),
        
        # Successful cancellation of ask order
        (
            [make_order(price=102, size=8, side="A", client_oid="ASK_1")],
            [],
            "ASK_1",
            ExecType.CANCELED,
            OrderStatus.CANCELED,
            True,
        ),
        
        # === MULTIPLE ORDERS TESTS ===
        
        # Cancel one order when multiple exist
        (
            [
                make_order(price=100, size=10, side="B", client_oid="ORDER_1"),
                make_order(price=102, size=15, side="A", client_oid="ORDER_2"),
            ],
            [],
            "ORDER_1",
            ExecType.CANCELED,
            OrderStatus.CANCELED,
            True,
        ),
        
        # Cancel second order when multiple exist
        (
            [
                make_order(price=100, size=10, side="B", client_oid="ORDER_1"),
                make_order(price=102, size=15, side="A", client_oid="ORDER_2"),
            ],
            [],
            "ORDER_2",
            ExecType.CANCELED,
            OrderStatus.CANCELED,
            True,
        ),
        
        # Cancel multiple orders sequentially
        (
            [
                make_order(price=100, size=10, side="B", client_oid="ORDER_1"),
                make_order(price=102, size=15, side="A", client_oid="ORDER_2"),
                make_order(price=99, size=20, side="B", client_oid="ORDER_3"),
            ],
            [],
            "ORDER_2",
            ExecType.CANCELED,
            OrderStatus.CANCELED,
            True,
        ),
        
        # === EDGE CASES ===
        
        # Cancel non-existent order
        (
            [make_order(price=100, size=10, side="B", client_oid="ORDER_1")],
            [],
            "NON_EXISTENT",
            ExecType.REJECTED,
            OrderStatus.REJECTED,
            False,
        ),
        
        # Cancel from empty exchange
        (
            [],
            [],
            "ANY_ORDER",
            ExecType.REJECTED,
            OrderStatus.REJECTED,
            False,
        ),
        
        # Cancel already cancelled order
        (
            [make_order(price=100, size=10, side="B", client_oid="ORDER_1")],
            [],
            "ORDER_1",
            ExecType.CANCELED,
            OrderStatus.CANCELED,
            True,
        ),
        
        # === ORDERS WITH MARKET DATA ===
        
        # Cancel order when market data exists
        (
            [make_order(price=100, size=10, side="B", client_oid="ORDER_1")],
            [
                make_market_data("Add", [
                    make_bid_ask_pair(100, 50, 102, 40),
                    make_bid_ask_pair(99, 30, 103, 35),
                ])
            ],
            "ORDER_1",
            ExecType.CANCELED,
            OrderStatus.CANCELED,
            True,
        ),
        
        # Cancel order after market update
        (
            [make_order(price=100, size=10, side="B", client_oid="ORDER_1")],
            [
                make_market_data("Add", [make_bid_ask_pair(100, 50, 102, 40)]),
                make_market_data("Modify", [make_bid_ask_pair(100, 30, 102, 40)]),
            ],
            "ORDER_1",
            ExecType.CANCELED,
            OrderStatus.CANCELED,
            True,
        ),
        
        # === ORDERS WITH DIFFERENT TYPES ===
        
        # Cancel limit order
        (
            [make_order(price=100, size=10, side="B", client_oid="LIMIT_1", order_type=OrderType.LIMIT)],
            [],
            "LIMIT_1",
            ExecType.CANCELED,
            OrderStatus.CANCELED,
            True,
        ),
        
        # Cancel order with different time in force
        (
            [make_order(price=100, size=10, side="B", client_oid="GTC_1", tif=TimeInForce.GTC)],
            [],
            "GTC_1",
            ExecType.CANCELED,
            OrderStatus.CANCELED,
            True,
        ),
        
        # === ORDERS WITH DIFFERENT SIZES ===
        
        # Cancel large order
        (
            [make_order(price=100, size=10000, side="B", client_oid="LARGE_1")],
            [],
            "LARGE_1",
            ExecType.CANCELED,
            OrderStatus.CANCELED,
            True,
        ),
        
        # Cancel small order
        (
            [make_order(price=100, size=1, side="B", client_oid="SMALL_1")],
            [],
            "SMALL_1",
            ExecType.CANCELED,
            OrderStatus.CANCELED,
            True,
        ),
        
        # === ORDERS AT DIFFERENT PRICES ===
        
        # Cancel order at high price
        (
            [make_order(price=1000000, size=10, side="B", client_oid="HIGH_1")],
            [],
            "HIGH_1",
            ExecType.CANCELED,
            OrderStatus.CANCELED,
            True,
        ),
        
        # Cancel order at low price
        (
            [make_order(price=1, size=10, side="B", client_oid="LOW_1")],
            [],
            "LOW_1",
            ExecType.CANCELED,
            OrderStatus.CANCELED,
            True,
        ),
        
        # Cancel order at zero price
        (
            [make_order(price=0, size=10, side="B", client_oid="ZERO_1")],
            [],
            "ZERO_1",
            ExecType.CANCELED,
            OrderStatus.CANCELED,
            True,
        ),
        
        # === ORDERS WITH AUTO-GENERATED CLIENT OIDS ===
        
        # Cancel order with auto-generated client OID
        (
            [make_order(price=100, size=10, side="B", client_oid=None)],  # Will be auto-generated
            [],
            None,  # We'll need to get the generated OID
            ExecType.CANCELED,
            OrderStatus.CANCELED,
            True,
        ),
        
        # === STRESS TESTS ===
        
        # Cancel order after many orders submitted
        (
            [make_order(price=100 + i, size=10, side="B", client_oid=f"ORDER_{i}") for i in range(100)],
            [],
            "ORDER_50",
            ExecType.CANCELED,
            OrderStatus.CANCELED,
            True,
        ),
        
        # Cancel order with special characters in client OID
        (
            [make_order(price=100, size=10, side="B", client_oid="ORDER_WITH_SPECIAL_CHARS_!@#$%")],
            [],
            "ORDER_WITH_SPECIAL_CHARS_!@#$%",
            ExecType.CANCELED,
            OrderStatus.CANCELED,
            True,
        ),
        
        # Cancel order with very long client OID
        (
            [make_order(price=100, size=10, side="B", client_oid="A" * 1000)],
            [],
            "A" * 1000,
            ExecType.CANCELED,
            OrderStatus.CANCELED,
            True,
        ),
    ]
)
def test_cancel_order(
    orders,
    market_data,
    client_oid_to_cancel,
    expected_exec_type,
    expected_order_status,
    expected_success,
):
    """Test cancel_order method with various scenarios"""
    exchange = setup_exchange_with_orders(orders, market_data)
    
    # Handle auto-generated client OID case
    if client_oid_to_cancel is None and orders:
        # Get the auto-generated client OID from the first order
        client_oid_to_cancel = orders[0].client_oid
    
    # Perform cancellation
    result = exchange.cancel_order(client_oid_to_cancel)
    
    # Verify result
    assert len(result) == 1
    execution_report = result[0]
    
    assert execution_report.client_oid == client_oid_to_cancel
    assert execution_report.exec_type == expected_exec_type
    assert execution_report.order_status == expected_order_status
    assert execution_report.filled_qty == 0
    assert execution_report.leaves_qty == 0
    
    # Verify order book state
    if expected_success:
        # Order should be removed from the order book
        assert client_oid_to_cancel not in exchange.order_book.local_orders['B']
        assert client_oid_to_cancel not in exchange.order_book.local_orders['A']
    else:
        # Order should not exist (was never there or already removed)
        pass


def test_cancel_order_multiple_cancellations():
    """Test cancelling the same order multiple times"""
    exchange = setup_exchange_with_orders([
        make_order(price=100, size=10, side="B", client_oid="ORDER_1")
    ])
    
    # First cancellation should succeed
    result1 = exchange.cancel_order("ORDER_1")
    assert len(result1) == 1
    assert result1[0].exec_type == ExecType.CANCELED
    assert result1[0].order_status == OrderStatus.CANCELED
    
    # Second cancellation should fail
    result2 = exchange.cancel_order("ORDER_1")
    assert len(result2) == 1
    assert result2[0].exec_type == ExecType.REJECTED
    assert result2[0].order_status == OrderStatus.REJECTED


def test_cancel_order_after_partial_fill():
    """Test cancelling an order that has been partially filled"""
    exchange = setup_exchange_with_orders([
        make_order(price=100, size=10, side="B", client_oid="ORDER_1")
    ])
    
    # Add market data to create liquidity
    market_data = make_market_data("Add", [
        make_bid_ask_pair(100, 50, 102, 40),
    ])
    exchange.on_market_data(market_data)
    
    # Simulate a partial fill (this would normally happen through on_market_data)
    # For this test, we'll manually modify the order to simulate partial fill
    local_order = exchange.order_book.local_orders['B']['ORDER_1']
    local_order.remaining = 5  # Simulate 5 units filled
    
    # Cancel the partially filled order
    result = exchange.cancel_order("ORDER_1")
    
    assert len(result) == 1
    assert result[0].exec_type == ExecType.CANCELED
    assert result[0].order_status == OrderStatus.CANCELED
    assert result[0].client_oid == "ORDER_1"


def test_cancel_order_empty_string():
    """Test cancelling with empty string client OID"""
    exchange = setup_exchange_with_orders([
        make_order(price=100, size=10, side="B", client_oid="ORDER_1")
    ])
    
    result = exchange.cancel_order("")
    
    assert len(result) == 1
    assert result[0].exec_type == ExecType.REJECTED
    assert result[0].order_status == OrderStatus.REJECTED
