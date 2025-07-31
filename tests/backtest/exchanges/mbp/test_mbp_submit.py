import pytest
from gnomepy.backtest.exchanges.mbp.mbp import MBPSimulatedExchange
from gnomepy.backtest.fee import FeeModel
from gnomepy.backtest.latency import LatencyModel
from gnomepy.data.types import (
    OrderType, TimeInForce, OrderStatus, ExecType,
    MBP10
)
from tests.backtest.exchanges.mbp.test_mbp_book import (
    DummyQueueModel, make_order, make_bid_ask_pair
)


class DummyFeeModel(FeeModel):
    def calculate_fee(self, notional: int, is_maker: bool) -> int:
        # 3% for maker, 5% for taker
        rate = 0.03 if is_maker else 0.05
        return int(notional * rate)


class DummyLatencyModel(LatencyModel):
    def simulate(self) -> int:
        return 0  # No latency for testing


def setup_exchange_with_market_data(market_data=None):
    """Helper to set up an exchange with market data"""
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
    "bid_ask_levels, order, expected_reports, expected_order_book_state, expect_error",
    [
        # === MARKET ORDER TESTS ===
        
        # Market buy order - full fill
        (
            [make_bid_ask_pair(100, 50, 102, 40)],
            make_order(price=0, size=20, side="B", order_type=OrderType.MARKET, client_oid="MARKET_BUY_1"),
            [
                {"exec_type": ExecType.TRADE, "order_status": OrderStatus.FILLED, "filled_qty": 20, "leaves_qty": 0, "filled_price": 107, "cumulative_qty": 20}
            ],
            {"bids": [100], "asks": [102], "local_orders": {}},
            False,
        ),

        # Market sell order - full fill
        (
            [make_bid_ask_pair(100, 50, 102, 40)],
            make_order(price=0, size=30, side="A", order_type=OrderType.MARKET, client_oid="MARKET_SELL_1"),
            [
                {"exec_type": ExecType.TRADE, "order_status": OrderStatus.FILLED, "filled_qty": 30, "leaves_qty": 0, "filled_price": 95, "cumulative_qty": 30}
            ],
            {"bids": [100], "asks": [102], "local_orders": {}},
            False,
        ),

        # Market order - no liquidity (should raise error)
        (
            [],
            make_order(price=0, size=10, side="B", order_type=OrderType.MARKET, client_oid="MARKET_NO_LIQUIDITY"),
            [],
            {},
            True,  # Should raise ValueError
        ),

        # Market order - insufficient liquidity (should raise error)
        (
            [make_bid_ask_pair(100, 5, 102, 5)],
            make_order(price=0, size=10, side="B", order_type=OrderType.MARKET, client_oid="MARKET_INSUFFICIENT"),
            [],
            {},
            True,  # Should raise ValueError
        ),

        # === LIMIT ORDER TESTS - NO MATCHES ===

        # Limit buy order - no matches, GTC (should be added to book)
        (
            [make_bid_ask_pair(100, 50, 102, 40)],
            make_order(price=99, size=10, side="B", order_type=OrderType.LIMIT, tif=TimeInForce.GTC, client_oid="LIMIT_BUY_GTC"),
            [
                {"exec_type": ExecType.NEW, "order_status": OrderStatus.NEW, "filled_qty": 0, "leaves_qty": 10, "filled_price": 0, "cumulative_qty": 0}
            ],
            {"bids": [100, 99], "asks": [102], "local_orders": {"B": ["LIMIT_BUY_GTC"]}},
            False,
        ),

        # Limit sell order - no matches, GTC (should be added to book)
        (
            [make_bid_ask_pair(100, 50, 102, 40)],
            make_order(price=103, size=10, side="A", order_type=OrderType.LIMIT, tif=TimeInForce.GTC, client_oid="LIMIT_SELL_GTC"),
            [
                {"exec_type": ExecType.NEW, "order_status": OrderStatus.NEW, "filled_qty": 0, "leaves_qty": 10, "filled_price": 0, "cumulative_qty": 0}
            ],
            {"bids": [100], "asks": [102, 103], "local_orders": {"A": ["LIMIT_SELL_GTC"]}},
            False,
        ),

        # Limit order - no matches, IOC (should be rejected)
        (
            [make_bid_ask_pair(100, 50, 102, 40)],
            make_order(price=99, size=10, side="B", order_type=OrderType.LIMIT, tif=TimeInForce.IOC, client_oid="LIMIT_BUY_IOC"),
            [
                {"exec_type": ExecType.REJECTED, "order_status": OrderStatus.REJECTED, "filled_qty": 0, "leaves_qty": 0, "filled_price": 0, "cumulative_qty": 0}
            ],
            {"bids": [100], "asks": [102], "local_orders": {}},
            False,
        ),

        # Limit order - no matches, FOK (should be rejected)
        (
            [make_bid_ask_pair(100, 50, 102, 40)],
            make_order(price=99, size=10, side="B", order_type=OrderType.LIMIT, tif=TimeInForce.FOK, client_oid="LIMIT_BUY_FOK"),
            [
                {"exec_type": ExecType.REJECTED, "order_status": OrderStatus.REJECTED, "filled_qty": 0, "leaves_qty": 0, "filled_price": 0, "cumulative_qty": 0}
            ],
            {"bids": [100], "asks": [102], "local_orders": {}},
            False,
        ),

        # === LIMIT ORDER TESTS - FULL FILLS ===

        # Limit buy order - full fill
        (
            [make_bid_ask_pair(100, 50, 102, 40)],
            make_order(price=102, size=40, side="B", order_type=OrderType.LIMIT, client_oid="LIMIT_BUY_FULL"),
            [
                {"exec_type": ExecType.TRADE, "order_status": OrderStatus.FILLED, "filled_qty": 40, "leaves_qty": 0, "filled_price": 107, "cumulative_qty": 40}
            ],
            {"bids": [100], "asks": [], "local_orders": {}},
            False,
        ),

        # Limit sell order - full fill
        (
            [make_bid_ask_pair(100, 50, 102, 40)],
            make_order(price=100, size=50, side="A", order_type=OrderType.LIMIT, client_oid="LIMIT_SELL_FULL"),
            [
                {"exec_type": ExecType.TRADE, "order_status": OrderStatus.FILLED, "filled_qty": 50, "leaves_qty": 0, "filled_price": 95, "cumulative_qty": 50}
            ],
            {"bids": [], "asks": [102], "local_orders": {}},
            False,
        ),

        # === LIMIT ORDER TESTS - PARTIAL FILLS ===

        # Limit buy order - partial fill, GTC (remaining should be added to book)
        (
            [make_bid_ask_pair(100, 50, 102, 40)],
            make_order(price=102, size=60, side="B", order_type=OrderType.LIMIT, tif=TimeInForce.GTC, client_oid="LIMIT_BUY_PARTIAL_GTC"),
            [
                {"exec_type": ExecType.NEW, "order_status": OrderStatus.NEW, "filled_qty": 0, "leaves_qty": 60, "filled_price": 0, "cumulative_qty": 0},
                {"exec_type": ExecType.TRADE, "order_status": OrderStatus.PARTIALLY_FILLED, "filled_qty": 40, "leaves_qty": 20, "filled_price": 107, "cumulative_qty": 40}
            ],
            {"bids": [102, 100], "asks": [], "local_orders": {"B": ["LIMIT_BUY_PARTIAL_GTC"]}},
            False,
        ),

        # Limit sell order - partial fill, GTC (remaining should be added to book)
        (
            [make_bid_ask_pair(100, 50, 102, 40)],
            make_order(price=100, size=60, side="A", order_type=OrderType.LIMIT, tif=TimeInForce.GTC, client_oid="LIMIT_SELL_PARTIAL_GTC"),
            [
                {"exec_type": ExecType.NEW, "order_status": OrderStatus.NEW, "filled_qty": 0, "leaves_qty": 60, "filled_price": 0, "cumulative_qty": 0},
                {"exec_type": ExecType.TRADE, "order_status": OrderStatus.PARTIALLY_FILLED, "filled_qty": 50, "leaves_qty": 10, "filled_price": 95, "cumulative_qty": 50}
            ],
            {"bids": [], "asks": [100, 102], "local_orders": {"A": ["LIMIT_SELL_PARTIAL_GTC"]}},
            False,
        ),

        # Limit buy order - partial fill, FOK (should be rejected)
        (
            [make_bid_ask_pair(100, 50, 102, 40)],
            make_order(price=102, size=60, side="B", order_type=OrderType.LIMIT, tif=TimeInForce.FOK, client_oid="LIMIT_BUY_PARTIAL_FOK"),
            [
                {"exec_type": ExecType.REJECTED, "order_status": OrderStatus.REJECTED, "filled_qty": 0, "leaves_qty": 0, "filled_price": 0, "cumulative_qty": 0}
            ],
            {"bids": [100], "asks": [102], "local_orders": {}},
            False,
        ),

        # Limit buy order - partial fill, IOC (should be cancelled)
        (
            [make_bid_ask_pair(100, 50, 102, 40)],
            make_order(price=102, size=60, side="B", order_type=OrderType.LIMIT, tif=TimeInForce.IOC, client_oid="LIMIT_BUY_PARTIAL_IOC"),
            [
                {"exec_type": ExecType.TRADE, "order_status": OrderStatus.PARTIALLY_FILLED, "filled_qty": 40, "leaves_qty": 20, "filled_price": 107, "cumulative_qty": 40},
                {"exec_type": ExecType.CANCELED, "order_status": OrderStatus.PARTIALLY_FILLED, "filled_qty": 0, "leaves_qty": 0, "filled_price": 0, "cumulative_qty": 40},
            ],
            {"bids": [100], "asks": [], "local_orders": {}},
            False,
        ),

        # === EDGE CASES ===

        # Order with auto-generated client OID
        (
            [make_bid_ask_pair(100, 50, 102, 40)],
            make_order(price=99, size=10, side="B", order_type=OrderType.LIMIT, client_oid=None),
            [
                {"exec_type": ExecType.NEW, "order_status": OrderStatus.NEW, "filled_qty": 0, "leaves_qty": 10, "filled_price": 0, "cumulative_qty": 0}
            ],
            {"bids": [100, 99], "asks": [102], "local_orders": {"B": ["auto_generated"]}},
            False,
        ),

                # Order with zero size
        (
            [make_bid_ask_pair(100, 50, 102, 40)],
            make_order(price=99, size=0, side="B", order_type=OrderType.LIMIT, client_oid="ZERO_SIZE"),
            [
                {"exec_type": ExecType.NEW, "order_status": OrderStatus.NEW, "filled_qty": 0, "leaves_qty": 0, "filled_price": 0, "cumulative_qty": 0}
            ],
            {"bids": [100, 99], "asks": [102], "local_orders": {"B": ["ZERO_SIZE"]}},
            False,
        ),
        
        # Order with very large size
        (
            [make_bid_ask_pair(100, 50, 102, 40)],
            make_order(price=99, size=1000000, side="B", order_type=OrderType.LIMIT, client_oid="LARGE_SIZE"),
            [
                {"exec_type": ExecType.NEW, "order_status": OrderStatus.NEW, "filled_qty": 0, "leaves_qty": 1000000, "filled_price": 0, "cumulative_qty": 0}
            ],
            {"bids": [100, 99], "asks": [102], "local_orders": {"B": ["LARGE_SIZE"]}},
            False,
        ),
        
        # Order at zero price
        (
            [make_bid_ask_pair(100, 50, 102, 40)],
            make_order(price=0, size=10, side="B", order_type=OrderType.LIMIT, client_oid="ZERO_PRICE"),
            [
                {"exec_type": ExecType.NEW, "order_status": OrderStatus.NEW, "filled_qty": 0, "leaves_qty": 10, "filled_price": 0, "cumulative_qty": 0}
            ],
            {"bids": [100, 0], "asks": [102], "local_orders": {"B": ["ZERO_PRICE"]}},
            False,
        ),
        
        # Order at very high price
        (
            [make_bid_ask_pair(100, 50, 102, 40)],
            make_order(price=1000000, size=10, side="A", order_type=OrderType.LIMIT, client_oid="HIGH_PRICE"),
            [
                {"exec_type": ExecType.NEW, "order_status": OrderStatus.NEW, "filled_qty": 0, "leaves_qty": 10, "filled_price": 0, "cumulative_qty": 0}
            ],
            {"bids": [100], "asks": [102, 1000000], "local_orders": {"A": ["HIGH_PRICE"]}},
            False,
        ),

        # Invalid order type
        (
            [],
            make_order(price=100, size=10, side="B", order_type="INVALID", client_oid="INVALID_TYPE"),
            [],
            {},
            True,  # Should raise ValueError
        ),

        # === MULTIPLE LEVEL MATCHING ===

        # Limit buy order matching multiple ask levels
        (
            [
                make_bid_ask_pair(100, 50, 102, 20),
                make_bid_ask_pair(100, 50, 103, 15),
                make_bid_ask_pair(100, 50, 104, 10),
            ],
            make_order(price=104, size=40, side="B", order_type=OrderType.LIMIT, client_oid="MULTI_LEVEL_BUY"),
            [
                {"exec_type": ExecType.TRADE, "order_status": OrderStatus.FILLED, "filled_qty": 40, "leaves_qty": 0, "filled_price": 107, "cumulative_qty": 40}
            ],
            {"bids": [100], "asks": [], "local_orders": {}},
            False,
        ),

        # Limit sell order matching multiple bid levels
        (
            [
                make_bid_ask_pair(100, 20, 102, 50),
                make_bid_ask_pair(99, 15, 102, 50),
                make_bid_ask_pair(98, 10, 102, 50),
            ],
            make_order(price=98, size=40, side="A", order_type=OrderType.LIMIT, client_oid="MULTI_LEVEL_SELL"),
            [
                {"exec_type": ExecType.TRADE, "order_status": OrderStatus.FILLED, "filled_qty": 40, "leaves_qty": 0, "filled_price": 94, "cumulative_qty": 40}
            ],
            {"bids": [], "asks": [102], "local_orders": {}},
            False,
        ),
        # === COMPLEX SCENARIOS ===

                # Order that creates new price level
        (
            [make_bid_ask_pair(100, 50, 102, 40)],
            make_order(price=95, size=10, side="B", order_type=OrderType.LIMIT, client_oid="NEW_LEVEL_BUY"),
            [
                {"exec_type": ExecType.NEW, "order_status": OrderStatus.NEW, "filled_qty": 0, "leaves_qty": 10, "filled_price": 0, "cumulative_qty": 0}
            ],
            {"bids": [100, 95], "asks": [102], "local_orders": {"B": ["NEW_LEVEL_BUY"]}},
            False,
        ),
        
        # Order that creates new ask level
        (
            [make_bid_ask_pair(100, 50, 102, 40)],
            make_order(price=105, size=10, side="A", order_type=OrderType.LIMIT, client_oid="NEW_LEVEL_SELL"),
            [
                {"exec_type": ExecType.NEW, "order_status": OrderStatus.NEW, "filled_qty": 0, "leaves_qty": 10, "filled_price": 0, "cumulative_qty": 0}
            ],
            {"bids": [100], "asks": [102, 105], "local_orders": {"A": ["NEW_LEVEL_SELL"]}},
            False,
        ),
    ]
)
def test_submit_order(
    bid_ask_levels,
    order,
    expected_reports,
    expected_order_book_state,
    expect_error,
):
    """Test submit_order method with various scenarios"""
    # Create market data with "Add" action
    market_data = [make_market_data("Add", bid_ask_levels)] if bid_ask_levels else []
    exchange = setup_exchange_with_market_data(market_data)
    
    if expect_error:
        with pytest.raises((ValueError, TypeError)):
            exchange.submit_order(order)
    else:
        result = exchange.submit_order(order)
        
        # Verify number of execution reports
        assert len(result) == len(expected_reports)
        
        # Verify each execution report
        for i, expected_report in enumerate(expected_reports):
            execution_report = result[i]
            
            # Handle auto-generated client OID case
            if expected_report.get("client_oid") == "auto_generated":
                assert execution_report.client_oid is not None
                assert execution_report.client_oid.startswith("client_")
            else:
                assert execution_report.client_oid == order.client_oid
            
            assert execution_report.exec_type == expected_report["exec_type"]
            assert execution_report.order_status == expected_report["order_status"]
            assert execution_report.filled_qty == expected_report["filled_qty"]
            assert execution_report.leaves_qty == expected_report["leaves_qty"]
            assert execution_report.filled_price == expected_report["filled_price"]
            assert execution_report.cumulative_qty == expected_report["cumulative_qty"]
        
        # Verify order book state
        if expected_order_book_state.get("bids"):
            assert exchange.order_book.bids == expected_order_book_state["bids"]
        
        if expected_order_book_state.get("asks"):
            assert exchange.order_book.asks == expected_order_book_state["asks"]
        
        # Verify local orders
        expected_local_orders = expected_order_book_state.get("local_orders", {})
        for side, expected_oids in expected_local_orders.items():
            if expected_oids == ["auto_generated"]:
                # Check that there's exactly one local order with auto-generated OID
                assert len(exchange.order_book.local_orders[side]) == 1
                actual_oid = list(exchange.order_book.local_orders[side].keys())[0]
                assert actual_oid.startswith("client_")
            else:
                for oid in expected_oids:
                    assert oid in exchange.order_book.local_orders[side]


def test_submit_order_duplicate_client_oid():
    """Test submitting orders with duplicate client OIDs"""
    exchange = setup_exchange_with_market_data([
        make_market_data("Add", [make_bid_ask_pair(100, 50, 102, 40)])
    ])
    
    # Submit first order
    order1 = make_order(price=99, size=10, side="B", order_type=OrderType.LIMIT, client_oid="DUPLICATE_OID")
    result1 = exchange.submit_order(order1)
    assert len(result1) == 1
    assert result1[0].exec_type == ExecType.NEW
    
    # Submit second order with same client OID
    order2 = make_order(price=98, size=5, side="B", order_type=OrderType.LIMIT, client_oid="DUPLICATE_OID")
    with pytest.raises(ValueError):
        exchange.submit_order(order2)


def test_submit_order_fee_calculation():
    """Test that fees are properly calculated in execution reports"""
    class TestFeeModel(FeeModel):
        def calculate_fee(self, notional: int, is_maker: bool) -> int:
            return notional // 100  # 1% fee
    
    exchange = MBPSimulatedExchange(
        fee_model=TestFeeModel(),
        network_latency=DummyLatencyModel(),
        order_processing_latency=DummyLatencyModel(),
        queue_model=DummyQueueModel(),
    )
    
    # Add market data
    market_data = make_market_data("Add", [make_bid_ask_pair(100, 50, 102, 40)])
    exchange.on_market_data(market_data)
    
    # Submit market buy order
    order = make_order(price=0, size=20, side="B", order_type=OrderType.MARKET, client_oid="FEE_TEST")
    result = exchange.submit_order(order)
    
    assert len(result) == 1
    execution_report = result[0]
    assert execution_report.exec_type == ExecType.TRADE
    assert execution_report.filled_qty == 20
    
    # Verify filled price includes fees (20 * 102 = 2040, 1% fee = 20, total = 2060, avg = 103)
    assert execution_report.filled_price == 103


def test_submit_order_auto_generated_oid_sequence():
    """Test that auto-generated client OIDs are unique and sequential"""
    exchange = setup_exchange_with_market_data()
    
    # Submit multiple orders without client OIDs
    orders = []
    for i in range(5):
        order = make_order(price=100 + i, size=10, side="B", order_type=OrderType.LIMIT, client_oid=None)
        result = exchange.submit_order(order)
        orders.append(result[0].client_oid)
    
    # Verify all OIDs are unique
    assert len(set(orders)) == 5
    
    # Verify OIDs follow the expected pattern
    for oid in orders:
        assert oid.startswith("client_")
        parts = oid.split("_")
        assert len(parts) == 3
        assert parts[1].isdigit()  # Counter should be numeric


def test_submit_order_edge_case_prices():
    """Test orders at extreme price levels"""
    exchange = setup_exchange_with_market_data([
        make_market_data("Add", [make_bid_ask_pair(100, 50, 102, 40)])
    ])
    
    # Test negative price (should work)
    order_negative = make_order(price=-100, size=10, side="B", order_type=OrderType.LIMIT, client_oid="NEGATIVE_PRICE")
    result_negative = exchange.submit_order(order_negative)
    assert len(result_negative) == 1
    assert result_negative[0].exec_type == ExecType.NEW
    
    # Test maximum integer price
    order_max = make_order(price=2**31 - 1, size=10, side="A", order_type=OrderType.LIMIT, client_oid="MAX_PRICE")
    result_max = exchange.submit_order(order_max)
    assert len(result_max) == 1
    assert result_max[0].exec_type == ExecType.NEW


def test_submit_order_edge_case_sizes():
    """Test orders with extreme sizes"""
    exchange = setup_exchange_with_market_data([
        make_market_data("Add", [make_bid_ask_pair(100, 50, 102, 40)])
    ])
    
    # Test maximum integer size
    order_max_size = make_order(price=99, size=2**31 - 1, side="B", order_type=OrderType.LIMIT, client_oid="MAX_SIZE")
    result_max_size = exchange.submit_order(order_max_size)
    assert len(result_max_size) == 1
    assert result_max_size[0].exec_type == ExecType.NEW
    
    # Test negative size (should work but might cause issues in real trading)
    order_negative_size = make_order(price=99, size=-10, side="B", order_type=OrderType.LIMIT, client_oid="NEGATIVE_SIZE")
    result_negative_size = exchange.submit_order(order_negative_size)
    assert len(result_negative_size) == 1
    assert result_negative_size[0].exec_type == ExecType.NEW

def test_submit_order_self_filling():
    """Test that self-filling is properly detected and prevented"""
    exchange = setup_exchange_with_market_data([
        make_market_data("Add", [make_bid_ask_pair(100, 50, 102, 40)])
    ])
    
    # Submit a buy order that would match the ask side
    buy_order = make_order(price=101, size=10, side="B", order_type=OrderType.LIMIT, client_oid="BUY_ORDER")
    buy_result = exchange.submit_order(buy_order)
    assert len(buy_result) == 1
    assert buy_result[0].exec_type == ExecType.NEW
    assert buy_result[0].order_status == OrderStatus.NEW
    
    # Now submit a sell order at the same price - this should trigger self-filling detection
    sell_order = make_order(price=101, size=10, side="A", order_type=OrderType.LIMIT, client_oid="SELL_ORDER")
    
    # This should raise a ValueError due to self-filling
    with pytest.raises(ValueError, match="Self filling triggered"):
        exchange.submit_order(sell_order)
