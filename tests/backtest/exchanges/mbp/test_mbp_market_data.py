from gnomepy import BidAskPair
from gnomepy.backtest.exchanges.mbp.mbp import MBPSimulatedExchange
from gnomepy.backtest.fee import FeeModel
from gnomepy.backtest.latency import LatencyModel
from gnomepy.data.types import (
    OrderType, OrderStatus, ExecType,
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
        return 0


def setup_exchange_with_local_orders():
    exchange = MBPSimulatedExchange(
        fee_model=DummyFeeModel(),
        network_latency=DummyLatencyModel(),
        order_processing_latency=DummyLatencyModel(),
        queue_model=DummyQueueModel(),
    )
    
    initial_market_data = make_market_data([make_bid_ask_pair(100, 50, 102, 40)])
    exchange.on_market_data(initial_market_data)
    
    return exchange

def make_market_data(levels: list[BidAskPair]) -> MBP10:
    return MBP10(
        action="Add",
        levels=levels,
        side=None,
        price=-1,
        size=-1,
        exchange_id=-1,
        security_id=-1,
        timestamp_sent=-1,
        timestamp_recv=-1,
        timestamp_event=-1,
        flags=[],
        sequence=-1,
        depth=-1,
    )


def make_trade_data(side: str, price: int, size: int) -> MBP10:
    return MBP10(
        action="Trade",
        levels=[],  # Ignored
        side=side,
        price=price,
        size=size,
        exchange_id=-1,
        security_id=-1,
        timestamp_sent=-1,
        timestamp_recv=-1,
        timestamp_event=-1,
        flags=[],
        sequence=-1,
        depth=-1,
    )


def test_on_market_data_trade_fills_local_order():
    """Test that real trade data properly fills local orders and generates correct execution reports"""
    exchange = setup_exchange_with_local_orders()
    
    # Add a local buy order at 101
    buy_order = make_order(price=101, size=50, side="B", order_type=OrderType.LIMIT, client_oid="BUY_ORDER")
    exchange.submit_order(buy_order)

    # Verify order is in the book
    assert "BUY_ORDER" in exchange.order_book.local_orders["B"]
    assert exchange.order_book.bids == [101, 100]
    
    # Simulate a real trade that should fill our local order
    trade_data = make_trade_data(side="A", price=101, size=30)  # Sell trade at 101
    execution_reports = exchange.on_market_data(trade_data)

    # Should get one execution report for the fill
    assert len(execution_reports) == 1
    report = execution_reports[0]

    # Verify execution report fields
    assert report.client_oid == "BUY_ORDER"
    assert report.exec_type == ExecType.TRADE
    assert report.order_status == OrderStatus.PARTIALLY_FILLED
    assert report.filled_qty == 30
    assert report.leaves_qty == 20  # 50 - 30 = 20
    assert report.cumulative_qty == 30

    # Verify filled price includes fees (maker fee for limit order)
    # 30 * 101 = 3030, 3% maker fee = 91, total = 3121, avg = 104
    expected_filled_price = 104
    assert report.filled_price == expected_filled_price

    # Verify order state in book
    local_order = exchange.order_book.local_orders["B"]["BUY_ORDER"]
    assert local_order.remaining == 20


def test_on_market_data_trade_fills_local_order_completely():
    """Test that a trade can completely fill a local order"""
    exchange = setup_exchange_with_local_orders()
    
    # Add a local buy order at 101
    buy_order = make_order(price=101, size=50, side="B", order_type=OrderType.LIMIT, client_oid="BUY_ORDER")
    exchange.submit_order(buy_order)

    # Simulate a trade that completely fills our order
    trade_data = make_trade_data(side="A", price=101, size=50)  # Sell trade at 101
    execution_reports = exchange.on_market_data(trade_data)
    
    # Should get one execution report
    assert len(execution_reports) == 1
    report = execution_reports[0]
    
    # Verify execution report
    assert report.client_oid == "BUY_ORDER"
    assert report.exec_type == ExecType.TRADE
    assert report.order_status == OrderStatus.FILLED
    assert report.filled_qty == 50
    assert report.leaves_qty == 0
    assert report.cumulative_qty == 50
    
    # Verify order is removed from book
    assert "BUY_ORDER" not in exchange.order_book.local_orders["B"]


def test_on_market_data_trade_multiple_partial_fills():
    """Test multiple trades that partially fill a local order over time"""
    exchange = setup_exchange_with_local_orders()
    
    # Add a large local buy order
    buy_order = make_order(price=101, size=100, side="B", order_type=OrderType.LIMIT, client_oid="LARGE_BUY")
    exchange.submit_order(buy_order)

    # First trade: fill 30 units
    trade1 = make_trade_data(side="A", price=101, size=30)
    reports1 = exchange.on_market_data(trade1)
    
    assert len(reports1) == 1
    report1 = reports1[0]
    assert report1.filled_qty == 30
    assert report1.leaves_qty == 70
    assert report1.cumulative_qty == 30
    assert report1.order_status == OrderStatus.PARTIALLY_FILLED
    
    # Second trade: fill 40 units
    trade2 = make_trade_data(side="A", price=101, size=40)
    reports2 = exchange.on_market_data(trade2)
    
    assert len(reports2) == 1
    report2 = reports2[0]
    assert report2.filled_qty == 40
    assert report2.leaves_qty == 30
    assert report2.cumulative_qty == 70  # 30 + 40
    assert report2.order_status == OrderStatus.PARTIALLY_FILLED
    
    # Third trade: fill remaining 30 units
    trade3 = make_trade_data(side="A", price=101, size=30)
    reports3 = exchange.on_market_data(trade3)
    
    assert len(reports3) == 1
    report3 = reports3[0]
    assert report3.filled_qty == 30
    assert report3.leaves_qty == 0
    assert report3.cumulative_qty == 100  # 30 + 40 + 30
    assert report3.order_status == OrderStatus.FILLED
    
    # Verify order is removed from book
    assert "LARGE_BUY" not in exchange.order_book.local_orders["B"]


def test_on_market_data_trade_no_match():
    """Test that trades that don't match local orders don't generate execution reports"""
    exchange = setup_exchange_with_local_orders()
    
    # Add a local buy order at 101
    buy_order = make_order(price=101, size=50, side="B", order_type=OrderType.LIMIT, client_oid="BUY_ORDER")
    exchange.submit_order(buy_order)

    # Simulate a trade at a different price (no match)
    trade_data = make_trade_data(side="A", price=102, size=30)  # Sell trade at 102
    execution_reports = exchange.on_market_data(trade_data)
    
    # Should get no execution reports
    assert len(execution_reports) == 0
    
    # Verify order is still in book unchanged
    local_order = exchange.order_book.local_orders["B"]["BUY_ORDER"]
    assert local_order.remaining == 50


def test_on_market_data_trade_multiple_local_orders():
    """Test that trades can fill multiple local orders at different price levels"""
    exchange = setup_exchange_with_local_orders()
    
    # Add multiple local buy orders at different prices
    buy_order1 = make_order(price=101, size=30, side="B", order_type=OrderType.LIMIT, client_oid="BUY_1")
    buy_order2 = make_order(price=100, size=40, side="B", order_type=OrderType.LIMIT, client_oid="BUY_2")
    exchange.submit_order(buy_order1)
    exchange.submit_order(buy_order2)

    # Simulate a large trade that fills both orders
    trade_data = make_trade_data(side="A", price=100, size=120)  # Large sell trade
    execution_reports = exchange.on_market_data(trade_data)
    
    # Should get two execution reports
    assert len(execution_reports) == 2
    
    # Find the reports for each order
    report1 = next(r for r in execution_reports if r.client_oid == "BUY_1")
    report2 = next(r for r in execution_reports if r.client_oid == "BUY_2")
    
    # Verify first order (at 101) is completely filled
    assert report1.filled_qty == 30
    assert report1.leaves_qty == 0
    assert report1.cumulative_qty == 30
    assert report1.order_status == OrderStatus.FILLED
    
    # Verify second order (at 100) is partially filled
    assert report2.filled_qty == 40
    assert report2.leaves_qty == 0
    assert report2.cumulative_qty == 40
    assert report2.order_status == OrderStatus.FILLED
    
    # Verify orders are removed from book
    assert "BUY_1" not in exchange.order_book.local_orders["B"]
    assert "BUY_2" not in exchange.order_book.local_orders["B"]


def test_on_market_data_bid_ask_update():
    """Test that bid/ask updates don't generate execution reports but update book state"""
    exchange = setup_exchange_with_local_orders()
    
    # Add a local buy order
    buy_order = make_order(price=101, size=50, side="B", order_type=OrderType.LIMIT, client_oid="BUY_ORDER")
    exchange.submit_order(buy_order)

    # Simulate a bid/ask update (not a trade)
    update_data = make_market_data([make_bid_ask_pair(99, 30, 103, 35)])
    execution_reports = exchange.on_market_data(update_data)
    
    # Should get no execution reports for bid/ask updates
    assert len(execution_reports) == 0
    
    # Verify order is still in book
    assert "BUY_ORDER" in exchange.order_book.local_orders["B"]
    local_order = exchange.order_book.local_orders["B"]["BUY_ORDER"]
    assert local_order.remaining == 50


def test_on_market_data_trade_fee_calculation():
    """Test that fees are properly calculated in execution reports from trades"""
    exchange = setup_exchange_with_local_orders()
    
    # Add a local buy order
    buy_order = make_order(price=101, size=50, side="B", order_type=OrderType.LIMIT, client_oid="BUY_ORDER")
    exchange.submit_order(buy_order)

    # Simulate a trade
    trade_data = make_trade_data(side="A", price=101, size=50)
    execution_reports = exchange.on_market_data(trade_data)
    
    assert len(execution_reports) == 1
    report = execution_reports[0]
    
    # Verify filled price includes maker fee
    # 50 * 101 = 5050, 3% maker fee = 152, total = 5202, avg = 104
    expected_filled_price = 104
    assert report.filled_price == expected_filled_price


def test_on_market_data_trade_sell_order():
    """Test that sell orders can be filled by buy trades"""
    exchange = setup_exchange_with_local_orders()
    
    # Add a local sell order at 102
    sell_order = make_order(price=102, size=40, side="A", order_type=OrderType.LIMIT, client_oid="SELL_ORDER")
    exchange.submit_order(sell_order)

    # Simulate a buy trade that should fill our sell order
    trade_data = make_trade_data(side="B", price=102, size=80)  # Buy trade at 102
    execution_reports = exchange.on_market_data(trade_data)
    
    # Should get one execution report
    assert len(execution_reports) == 1
    report = execution_reports[0]
    
    # Verify execution report
    assert report.client_oid == "SELL_ORDER"
    assert report.exec_type == ExecType.TRADE
    assert report.order_status == OrderStatus.FILLED
    assert report.filled_qty == 40
    assert report.leaves_qty == 0
    assert report.cumulative_qty == 40
    
    # Verify order is removed from book
    assert "SELL_ORDER" not in exchange.order_book.local_orders["A"]


def test_on_market_data_trade_price_matching():
    """Test that trades only match local orders at the correct price levels"""
    exchange = setup_exchange_with_local_orders()
    
    # Add local orders at different prices
    buy_order1 = make_order(price=101, size=30, side="B", order_type=OrderType.LIMIT, client_oid="BUY_101")
    buy_order2 = make_order(price=100, size=30, side="B", order_type=OrderType.LIMIT, client_oid="BUY_100")
    exchange.submit_order(buy_order1)
    exchange.submit_order(buy_order2)

    # Simulate a trade at 101 - should only fill the 101 order
    trade_data = make_trade_data(side="A", price=101, size=30)
    execution_reports = exchange.on_market_data(trade_data)
    
    # Should get one execution report
    assert len(execution_reports) == 1
    report = execution_reports[0]
    assert report.client_oid == "BUY_101"
    assert report.filled_qty == 30
    assert report.order_status == OrderStatus.FILLED
    
    # Verify 101 order is removed, 100 order remains
    assert "BUY_101" not in exchange.order_book.local_orders["B"]
    assert "BUY_100" in exchange.order_book.local_orders["B"]
    assert exchange.order_book.local_orders["B"]["BUY_100"].remaining == 30

def test_on_market_data_trade_lifecycle_complete():
    """Test complete lifecycle of a local order through multiple trades"""
    exchange = setup_exchange_with_local_orders()
    
    # Add a large local buy order
    buy_order = make_order(price=101, size=100, side="B", order_type=OrderType.LIMIT, client_oid="LIFECYCLE_BUY")
    exchange.submit_order(buy_order)

    # Phase 1: Initial state
    local_order = exchange.order_book.local_orders["B"]["LIFECYCLE_BUY"]
    assert local_order.remaining == 100
    
    # Phase 2: First partial fill (30 units)
    trade1 = make_trade_data(side="A", price=101, size=30)
    reports1 = exchange.on_market_data(trade1)
    assert len(reports1) == 1
    assert reports1[0].filled_qty == 30
    assert reports1[0].cumulative_qty == 30
    assert reports1[0].order_status == OrderStatus.PARTIALLY_FILLED
    
    local_order = exchange.order_book.local_orders["B"]["LIFECYCLE_BUY"]
    assert local_order.remaining == 70
    
    # Phase 3: Second partial fill (40 units)
    trade2 = make_trade_data(side="A", price=101, size=40)
    reports2 = exchange.on_market_data(trade2)
    assert len(reports2) == 1
    assert reports2[0].filled_qty == 40
    assert reports2[0].cumulative_qty == 70
    assert reports2[0].order_status == OrderStatus.PARTIALLY_FILLED
    
    local_order = exchange.order_book.local_orders["B"]["LIFECYCLE_BUY"]
    assert local_order.remaining == 30
    
    # Phase 4: Final fill (30 units)
    trade3 = make_trade_data(side="A", price=101, size=30)
    reports3 = exchange.on_market_data(trade3)
    assert len(reports3) == 1
    assert reports3[0].filled_qty == 30
    assert reports3[0].cumulative_qty == 100
    assert reports3[0].order_status == OrderStatus.FILLED
    
    # Phase 5: Order should be removed from book
    assert "LIFECYCLE_BUY" not in exchange.order_book.local_orders["B"] 