import pytest
from gnomepy.backtest.exchanges.mbp.mbp_book import MBPBook, OrderBookLevel
from gnomepy.data.types import MBP10
from tests.backtest.exchanges.mbp.test_mbp_book import DummyQueueModel, make_order

def make_trade(
    price,
    size,
    side,
    exchange_id=1,
    security_id=1,
    timestamp_event=1000000000000,
    timestamp_sent=None,
    timestamp_recv=1000000000000,
    action="T",
    flags=None,
    sequence=None,
    depth=None,
):
    if flags is None:
        flags = []
    return MBP10(
        exchange_id=exchange_id,
        security_id=security_id,
        timestamp_event=timestamp_event,
        timestamp_sent=timestamp_sent,
        timestamp_recv=timestamp_recv,
        price=price,
        size=size,
        action=action,
        side=side,
        flags=flags,
        sequence=sequence,
        depth=depth,
        levels=[],  # Not used in on_trade
    )

def setup_book_with_local_orders_and_market_volume(
    local_orders,
    market_bids=None,
    market_asks=None,
):
    """Setup book with local orders and existing market volume."""
    book = MBPBook(queue_model=DummyQueueModel())
    
    # Add market volume first
    if market_bids:
        for price, size in market_bids:
            level = OrderBookLevel(price=price, size=size)
            book.price_to_level['B'][price] = level
            book.bids.append(price)
        book.bids.sort(reverse=True)
    
    if market_asks:
        for price, size in market_asks:
            level = OrderBookLevel(price=price, size=size)
            book.price_to_level['A'][price] = level
            book.asks.append(price)
        book.asks.sort()
    
    for order in local_orders:
        book.add_local_order(order)
    
    return book

@pytest.mark.parametrize(
    "local_orders, market_bids, market_asks, trade, expected_fills",
    [
        # No local orders - should return empty
        (
            [],
            [(100, 10)],
            [],
            make_trade(price=100, size=5, side="A"),
            [],
        ),
        # Sell trade at better price than best bid - should get fill after consuming market volume
        (
            [make_order(price=100, size=5, side="B", client_oid="A")],
            [(100, 3)],  # 3 units of existing market volume
            [],
            make_trade(price=99, size=8, side="A"),  # sell at 99, best bid is 100 - GOOD PRICE
            [("A", 5)],  # Should get filled after consuming 3 market units
        ),
        # Sell trade at worse price than best bid - no fill
        (
            [make_order(price=100, size=5, side="B", client_oid="A")],
            [(100, 10)],
            [],
            make_trade(price=101, size=5, side="A"),  # sell at 101, best bid is 100 - BAD PRICE
            [],
        ),
        # Buy trade at better price than best ask - should get fill after consuming market volume
        (
            [make_order(price=102, size=5, side="A", client_oid="A")],
            [],
            [(102, 4)],  # 4 units of existing market volume
            make_trade(price=103, size=9, side="B"),  # buy at 103, best ask is 102 - GOOD PRICE
            [("A", 5)],  # Should get filled after consuming 4 market units
        ),
        # Buy trade at worse price than best ask - no fill
        (
            [make_order(price=102, size=5, side="A", client_oid="A")],
            [],
            [(102, 10)],
            make_trade(price=101, size=5, side="B"),  # buy at 101, best ask is 102 - BAD PRICE
            [],
        ),
        # Trade matches local order exactly but gets consumed by existing market volume
        (
            [make_order(price=100, size=5, side="B", client_oid="A")],
            [(100, 10)],
            [],
            make_trade(price=100, size=5, side="A"),
            [],  # No local fill, all consumed by existing market volume
        ),
        # Trade partially fills local order after consuming existing market volume
        (
            [make_order(price=100, size=10, side="B", client_oid="A")],
            [(100, 5)],  # 5 units of existing market volume
            [],
            make_trade(price=100, size=8, side="A"),
            [("A", 3)],  # 3 units fill local order after consuming 5 market units
        ),
        # Trade larger than local order after consuming existing market volume
        (
            [make_order(price=100, size=5, side="B", client_oid="A")],
            [(100, 3)],  # 3 units of existing market volume
            [],
            make_trade(price=100, size=10, side="A"),
            [("A", 5)],  # Local order gets filled after consuming 3 market units
        ),
        # Trade fills multiple local orders at same price after consuming existing market volume
        (
            [
                make_order(price=100, size=3, side="B", client_oid="A"),
                make_order(price=100, size=2, side="B", client_oid="B"),
            ],
            [(100, 1)],  # 1 unit of existing market volume
            [],
            make_trade(price=100, size=6, side="A"),
            [("A", 3), ("B", 2)],  # Both orders filled after consuming 1 market unit
        ),
        # Trade fills orders at multiple price levels after consuming existing market volume
        (
            [
                make_order(price=100, size=3, side="B", client_oid="A"),
                make_order(price=99, size=2, side="B", client_oid="B"),
            ],
            [(100, 2), (99, 1)],  # 2 units at 100, 1 unit at 99
            [],
            make_trade(price=99, size=8, side="A"),
            [("A", 3), ("B", 2)],  # Both orders filled after consuming market volume
        ),
        # Trade with existing market volume - local order gets filled after consuming market volume
        (
            [make_order(price=100, size=5, side="B", client_oid="A")],
            [(100, 10)],  # 10 units of existing market volume
            [],
            make_trade(price=100, size=15, side="A"),
            [("A", 5)],  # Local order gets filled after consuming 10 market units
        ),
        # Trade exactly consumes existing market volume - no local fill
        (
            [make_order(price=100, size=5, side="B", client_oid="A")],
            [(100, 5)],  # 5 units of existing market volume
            [],
            make_trade(price=100, size=5, side="A"),
            [],  # No local fill, all consumed by existing market volume
        ),
        # Trade smaller than existing market volume - no local fill
        (
            [make_order(price=100, size=5, side="B", client_oid="A")],
            [(100, 10)],  # 10 units of existing market volume
            [],
            make_trade(price=100, size=3, side="A"),
            [],  # No local fill, all consumed by existing market volume
        ),
        # Trade with multiple orders and existing market volume
        (
            [
                make_order(price=100, size=3, side="B", client_oid="A"),
                make_order(price=100, size=2, side="B", client_oid="B"),
            ],
            [(100, 4)],  # 4 units of existing market volume
            [],
            make_trade(price=100, size=10, side="A"),
            [("A", 3), ("B", 2)],  # Both orders filled after consuming 4 market units
        ),
        # Ask side trade filling bid orders after consuming market volume
        (
            [make_order(price=100, size=5, side="B", client_oid="A")],
            [(100, 2)],  # 2 units of existing market volume
            [],
            make_trade(price=100, size=7, side="A"),
            [("A", 5)],  # Local order gets filled after consuming 2 market units
        ),
        # Bid side trade filling ask orders after consuming market volume
        (
            [make_order(price=102, size=5, side="A", client_oid="A")],
            [],
            [(102, 1)],  # 1 unit of existing market volume
            make_trade(price=102, size=6, side="B"),
            [("A", 5)],  # Local order gets filled after consuming 1 market unit
        ),
        # Trade at different price than local orders - no fill
        (
            [make_order(price=100, size=5, side="B", client_oid="A")],
            [(100, 10)],
            [],
            make_trade(price=101, size=5, side="A"),  # trade at 101, order at 100
            [],
        ),
        # Trade with zero size
        (
            [make_order(price=100, size=5, side="B", client_oid="A")],
            [(100, 10)],
            [],
            make_trade(price=100, size=0, side="A"),
            [],
        ),
        # Sell trade at better price with existing market volume - should get fill after consuming market volume
        (
            [make_order(price=100, size=5, side="B", client_oid="A")],
            [(100, 8)],  # 8 units of existing market volume
            [],
            make_trade(price=99, size=15, side="A"),  # sell at 99, best bid is 100 - GOOD PRICE
            [("A", 5)],  # Should get filled after consuming 8 market units
        ),
        # Buy trade at better price with existing market volume - should get fill after consuming market volume
        (
            [make_order(price=102, size=5, side="A", client_oid="A")],
            [],
            [(102, 7)],  # 7 units of existing market volume
            make_trade(price=103, size=15, side="B"),  # buy at 103, best ask is 102 - GOOD PRICE
            [("A", 5)],  # Should get filled after consuming 7 market units
        ),
    ]
)
def test_on_trade(
    local_orders,
    market_bids,
    market_asks,
    trade,
    expected_fills,
):
    book = setup_book_with_local_orders_and_market_volume(
        local_orders=local_orders,
        market_bids=market_bids,
        market_asks=market_asks,
    )
    
    fills = book.on_trade(trade)
    
    # Convert fills to (client_oid, size) tuples for comparison
    actual_fills = [(local_order.order.client_oid, size) for local_order, size in fills]
    assert actual_fills == expected_fills

@pytest.mark.parametrize(
    "local_orders, market_bids, market_asks, trade, expect_malformed_error",
    [
        # Malformed book - price level exists but no OrderBookLevel
        (
            [make_order(price=100, size=5, side="B", client_oid="A")],
            [(100, 10)],
            [],
            make_trade(price=100, size=5, side="A"),
            True,  # Should raise error due to malformed book
        ),
    ]
)
def test_on_trade_malformed_book(
    local_orders,
    market_bids,
    market_asks,
    trade,
    expect_malformed_error,
):
    book = setup_book_with_local_orders_and_market_volume(
        local_orders=local_orders,
        market_bids=market_bids,
        market_asks=market_asks,
    )
    
    # Manually create malformed state by removing OrderBookLevel
    if trade.side == "A":  # sell trade, so opposite side is bids
        opp_side = "B"
        opp_list = book.bids
    else:  # buy trade, so opposite side is asks
        opp_side = "A"
        opp_list = book.asks
    
    if opp_list:
        price = opp_list[0]
        book.price_to_level[opp_side].pop(price, None)
    
    if expect_malformed_error:
        with pytest.raises(ValueError, match="Malformed local book"):
            book.on_trade(trade)
    else:
        book.on_trade(trade)

def test_on_trade_empty_book():
    """Test on_trade with completely empty book."""
    book = MBPBook(queue_model=DummyQueueModel())
    trade = make_trade(price=100, size=5, side="A")
    
    fills = book.on_trade(trade)
    assert fills == []

def test_on_trade_no_local_orders():
    """Test on_trade with market volume but no local orders."""
    book = MBPBook(queue_model=DummyQueueModel())
    
    # Add market volume without local orders
    level = OrderBookLevel(price=100, size=10)
    book.price_to_level['B'][100] = level
    book.bids.append(100)
    
    trade = make_trade(price=100, size=5, side="A")
    fills = book.on_trade(trade)
    
    assert fills == []

def test_on_trade_price_boundary_conditions():
    """Test on_trade with various price boundary conditions."""
    book = MBPBook(queue_model=DummyQueueModel())
    
    # Add local order at price 100
    order = make_order(price=100, size=5, side="B", client_oid="A")
    book.add_local_order(order)
    
    # Add market volume
    level = OrderBookLevel(price=100, size=10)
    book.price_to_level['B'][100].size = 10

    # Test trade exactly at order price
    trade1 = make_trade(price=100, size=4, side="A")
    fills1 = book.on_trade(trade1)
    assert [(lo.order.client_oid, size) for lo, size in fills1] == [("A", 4)]
    
    # Test trade one tick below (should not match)
    trade2 = make_trade(price=101, size=5, side="A")
    fills2 = book.on_trade(trade2)
    assert fills2 == []

def test_on_trade_multiple_iterations_order_removal():
    """Test that orders get properly removed after being filled and don't get filled twice."""
    book = MBPBook(queue_model=DummyQueueModel())
    
    # Add local orders
    order1 = make_order(price=100, size=5, side="B", client_oid="A")
    order2 = make_order(price=100, size=3, side="B", client_oid="B")
    order3 = make_order(price=99, size=4, side="B", client_oid="C")
    
    book.add_local_order(order1)
    book.add_local_order(order2)
    book.add_local_order(order3)
    
    # Add market volume
    book.price_to_level['B'][100].size = 4
    book.price_to_level['B'][99].size = 2

    # First trade: should fill order A completely and order B partially
    trade1 = make_trade(price=100, size=6, side="A")
    fills1 = book.on_trade(trade1)
    
    # Verify first trade fills
    assert [(lo.order.client_oid, size) for lo, size in fills1] == [("A", 5), ("B", 1)]
    
    # Verify order A is removed from local_orders
    assert "A" not in book.local_orders['B']
    assert "B" in book.local_orders['B']
    assert "C" in book.local_orders['B']

    # Verify order A is removed from price level deque
    assert len(book.price_to_level['B'][100].local_orders) == 1
    assert book.price_to_level['B'][100].local_orders[0].order.client_oid == "B"

    # Second trade: should fill remaining part of order B and some of the resting volume
    trade2 = make_trade(price=100, size=5, side="A")
    fills2 = book.on_trade(trade2)

    # Verify second trade fills
    assert [(lo.order.client_oid, size) for lo, size in fills2] == [("B", 2)]

    # Verify order B is now removed
    assert "B" not in book.local_orders['B']
    assert "C" in book.local_orders['B']

    # Verify order B is removed from price level deque
    assert len(book.price_to_level['B'][100].local_orders) == 0
    assert book.price_to_level['B'][100].size == 1

    # Third trade: should fill order C at different price level
    trade3 = make_trade(price=99, size=5, side="A")
    fills3 = book.on_trade(trade3)

    # Verify third trade fills
    assert [(lo.order.client_oid, size) for lo, size in fills3] == [("C", 4)]

    # Verify order C is removed
    assert "C" not in book.local_orders['B']
    assert len(book.local_orders['B']) == 0
    assert book.price_to_level['B'][100].size == 0

    # Verify order C is removed from price level deque
    assert len(book.price_to_level['B'][99].local_orders) == 0

    # Fourth trade: should return no fills since all orders are gone
    trade4 = make_trade(price=100, size=10, side="A")
    fills4 = book.on_trade(trade4)

    # Verify no fills
    assert fills4 == []

    # Verify no local orders remain
    assert len(book.local_orders['B']) == 0
    assert len(book.local_orders['A']) == 0

def test_on_trade_extensive_queue_position_logic():
    """
    Extensive test of queue position logic with multiple volume levels, 
    multiple local orders, and multiple sequential trades.
    
    This test verifies:
    1. Existing market volume is consumed before local orders
    2. Local orders are filled in queue order (FIFO)
    3. Market volume levels are consumed correctly
    4. Orders are properly removed after being filled
    5. Complex scenarios with multiple price levels
    6. Phantom volume is correctly assigned to local orders
    """
    book = MBPBook(queue_model=DummyQueueModel())
    
    # Setup complex book with existing market volume BEFORE adding local orders
    # This creates phantom volume for local orders
    
    # Create price levels with existing market volume first
    # Bid side: 100 (market: 15), 99 (market: 8), 98 (market: 12)
    # Ask side: 102 (market: 10), 103 (market: 6)
    
    # Add existing market volume to create price levels
    bid_levels = [
        (100, 15),  # 15 units of existing market volume
        (99, 8),    # 8 units of existing market volume  
        (98, 12),   # 12 units of existing market volume
    ]
    
    ask_levels = [
        (102, 10),  # 10 units of existing market volume
        (103, 6),   # 6 units of existing market volume
    ]
    
    # Create bid levels with existing market volume
    for price, size in bid_levels:
        level = OrderBookLevel(price=price, size=size)
        book.price_to_level['B'][price] = level
        book.bids.append(price)
    book.bids.sort(reverse=True)
    
    # Create ask levels with existing market volume
    for price, size in ask_levels:
        level = OrderBookLevel(price=price, size=size)
        book.price_to_level['A'][price] = level
        book.asks.append(price)
    book.asks.sort()
    
    # Now add local orders - this will create phantom volume
    # Bid side: 100 (locals: A=5, B=3), 99 (locals: C=4, D=2), 98 (locals: E=6)
    # Ask side: 102 (locals: F=3, G=2), 103 (locals: H=4)
    orders = [
        make_order(price=100, size=5, side="B", client_oid="A"),
        make_order(price=100, size=3, side="B", client_oid="B"),
        make_order(price=99, size=4, side="B", client_oid="C"),
        make_order(price=99, size=2, side="B", client_oid="D"),
        make_order(price=98, size=6, side="B", client_oid="E"),
        make_order(price=97, size=15, side="B", client_oid="F"),
        make_order(price=102, size=3, side="A", client_oid="G"),
        make_order(price=102, size=2, side="A", client_oid="H"),
        make_order(price=103, size=4, side="A", client_oid="I"),
    ]
    
    for order in orders:
        book.add_local_order(order)
    
    # Verify phantom volume is correctly assigned
    # Order A should have phantom_volume=15 (existing market volume at 100)
    assert book.local_orders['B']['A'].phantom_volume == 15
    # Order B should have phantom_volume=15 (existing market volume at 100)
    assert book.local_orders['B']['B'].phantom_volume == 15
    # Order C should have phantom_volume=8 (existing market volume at 99)
    assert book.local_orders['B']['C'].phantom_volume == 8
    # Order D should have phantom_volume=8 (existing market volume at 99)
    assert book.local_orders['B']['D'].phantom_volume == 8
    # Order E should have phantom_volume=12 (existing market volume at 98)
    assert book.local_orders['B']['E'].phantom_volume == 12
    # Order F should have phantom_volume=0 (existing market volume at 0)
    assert book.local_orders['B']['F'].phantom_volume == 0
    # Order G should have phantom_volume=10 (existing market volume at 102)
    assert book.local_orders['A']['G'].phantom_volume == 10
    # Order H should have phantom_volume=10 (existing market volume at 102)
    assert book.local_orders['A']['H'].phantom_volume == 10
    # Order I should have phantom_volume=6 (existing market volume at 103)
    assert book.local_orders['A']['I'].phantom_volume == 6

    # Verify initial state
    assert len(book.local_orders['B']) == 6  # A, B, C, D, E, F
    assert len(book.local_orders['A']) == 3  # G, H, I
    assert book.price_to_level['B'][100].size == 15
    assert book.price_to_level['B'][99].size == 8
    assert book.price_to_level['B'][98].size == 12
    assert book.price_to_level['B'][97].size == 0
    assert book.price_to_level['A'][102].size == 10
    assert book.price_to_level['A'][103].size == 6

    # === SEQUENCE 1: Sell trades against bid side ===

    # Trade 1: Sell 10 units at 100 - should consume 10 market units, no local fills
    # Phantom volume for orders A and B should be reduced from 15 to 5
    trade1 = make_trade(price=100, size=10, side="A")
    fills1 = book.on_trade(trade1)
    assert fills1 == []  # No local fills, all consumed by market volume
    assert book.price_to_level['B'][100].size == 5  # 15 - 10 = 5 remaining
    assert len(book.local_orders['B']) == 6  # All orders still present

    # Verify phantom volume was consumed for orders at price 100
    assert book.local_orders['B']['A'].phantom_volume == 5  # 15 - 10 = 5 remaining
    assert book.local_orders['B']['B'].phantom_volume == 5  # 15 - 10 = 5 remaining
    # Orders at other prices should have unchanged phantom volume
    assert book.local_orders['B']['C'].phantom_volume == 8  # Unchanged
    assert book.local_orders['B']['D'].phantom_volume == 8  # Unchanged
    assert book.local_orders['B']['E'].phantom_volume == 12  # Unchanged
    assert book.local_orders['B']['F'].phantom_volume == 0  # Unchanged

    # Trade 2: Sell 8 units at 100 - should consume 5 remaining market units + 3 from order A
    # Phantom volume for order B should be reduced from 5 to 0, and order A gets filled
    trade2 = make_trade(price=100, size=10, side="A")
    fills2 = book.on_trade(trade2)
    assert [(lo.order.client_oid, size) for lo, size in fills2] == [("A", 5)]  # Order A gets 5 units filled
    assert book.price_to_level['B'][100].size == 0  # All market volume consumed
    assert "A" not in book.local_orders['B']  # Order A removed (fully filled)
    assert "B" in book.local_orders['B']  # Order B still present
    assert len(book.local_orders['B']) == 5  # B, C, D, E, F remaining

    # Verify phantom volume was consumed for order B
    assert book.local_orders['B']['B'].phantom_volume <= 0  # 5 - 5 = 0 remaining
    # Orders at other prices should have unchanged phantom volume
    assert book.local_orders['B']['C'].phantom_volume == 8  # Unchanged
    assert book.local_orders['B']['D'].phantom_volume == 8  # Unchanged
    assert book.local_orders['B']['E'].phantom_volume == 12  # Unchanged
    assert book.local_orders['B']['F'].phantom_volume == 0  # Unchanged

    # Trade 3: Sell 5 units at 100 - should fill order B completely and some market volume
    trade3 = make_trade(price=99, size=5, side="A")
    fills3 = book.on_trade(trade3)
    assert [(lo.order.client_oid, size) for lo, size in fills3] == [("B", 3)]  # Order B gets filled
    assert book.price_to_level['B'][99].size == 6
    assert "B" not in book.local_orders['B']  # Order B removed
    assert len(book.local_orders['B']) == 4  # C, D, E, F remaining

    # Trade 4: Sell 20 units at 99 - should consume 8 market units + 4 from order C + 2 from order D + 6 from order E
    trade4 = make_trade(price=98, size=20, side="A")
    fills4 = book.on_trade(trade4)
    assert [(lo.order.client_oid, size) for lo, size in fills4] == [("C", 4), ("D", 2)]  # All orders at 99 filled
    assert book.price_to_level['B'][99].size == 0  # All market volume consumed
    assert book.price_to_level['B'][98].size == 4  # All market volume consumed
    assert "C" not in book.local_orders['B']  # All orders removed
    assert "D" not in book.local_orders['B']
    assert "E" in book.local_orders['B']
    assert book.local_orders['B']['E'].phantom_volume == 4
    assert len(book.local_orders['B']) == 2

    # === SEQUENCE 2: Buy trades against ask side ===

    # Trade 5: Buy 8 units at 102 - should consume 8 market units, no local fills
    # Phantom volume for orders F and G should be reduced from 10 to 2
    trade5 = make_trade(price=102, size=8, side="B")
    fills5 = book.on_trade(trade5)
    assert fills5 == []  # No local fills, all consumed by market volume
    assert book.price_to_level['A'][102].size == 2  # 10 - 8 = 2 remaining
    assert len(book.local_orders['A']) == 3  # All orders still present

    # Verify phantom volume was consumed for orders at price 102
    assert book.local_orders['A']['G'].phantom_volume == 2  # 10 - 8 = 2 remaining
    assert book.local_orders['A']['H'].phantom_volume == 2  # 10 - 8 = 2 remaining
    # Order at different price should have unchanged phantom volume
    assert book.local_orders['A']['I'].phantom_volume == 6  # Unchanged

    # Trade 6: Buy 5 units at 102 - should consume 2 remaining market units + 3 from order F
    trade6 = make_trade(price=102, size=5, side="B")
    fills6 = book.on_trade(trade6)
    assert [(lo.order.client_oid, size) for lo, size in fills6] == [("G", 3)]  # Order G gets filled
    assert book.price_to_level['A'][102].size == 0  # All market volume consumed
    assert "G" not in book.local_orders['A']  # Order F removed
    assert "H" in book.local_orders['A']  # Order G still present
    assert len(book.local_orders['A']) == 2  # H, I remaining

    # Trade 7: Buy 3 units at 103 - should fill order G completely
    trade7 = make_trade(price=103, size=3, side="B")
    fills7 = book.on_trade(trade7)
    assert [(lo.order.client_oid, size) for lo, size in fills7] == [("H", 2)]  # Order H gets filled
    assert "H" not in book.local_orders['A']  # Order G removed
    assert book.price_to_level['A'][103].size == 5
    assert len(book.local_orders['A']) == 1  # H remaining

    # Trade 8: Buy 12 units at 103 - should consume 6 market units + 4 from order H
    trade8 = make_trade(price=103, size=12, side="B")
    fills8 = book.on_trade(trade8)
    assert [(lo.order.client_oid, size) for lo, size in fills8] == [("I", 4)]  # Order I gets filled
    assert book.price_to_level['A'][103].size == 0  # All market volume consumed
    assert "I" not in book.local_orders['A']  # Order H removed
    assert len(book.local_orders['A']) == 0  # No ask orders remaining

    # === SEQUENCE 3: Edge cases and boundary conditions ===

    # Trade 9: Try to trade against empty book - should return no fills
    trade9 = make_trade(price=100, size=10, side="A")
    fills9 = book.on_trade(trade9)
    assert fills9 == []  # No fills, no orders left

    # Trade 10: Try to trade at different price - clear out the order book
    trade10 = make_trade(price=95, size=25, side="A")
    fills10 = book.on_trade(trade10)
    assert [(lo.order.client_oid, size) for lo, size in fills10] == [("E", 6), ("F", 15)]  # Order F gets filled

    # === SEQUENCE 4: Test price priority (better prices get filled first) ===

    # Add new orders at different prices
    new_orders = [
        make_order(price=101, size=3, side="B", client_oid="I"),  # Better price than 100
        make_order(price=100, size=2, side="B", client_oid="J"),  # Worse price than 101
    ]

    for order in new_orders:
        book.add_local_order(order)

    # Add market volume
    book.price_to_level['B'][101].size = 5
    book.price_to_level['B'][100].size = 3

    # Trade 11: Sell 10 units at 101 - should fill order I first (better price)
    trade11 = make_trade(price=101, size=10, side="A")
    fills11 = book.on_trade(trade11)
    assert [(lo.order.client_oid, size) for lo, size in fills11] == [("I", 3)]  # Order I gets filled first
    assert book.price_to_level['B'][101].size == 0
    assert "I" not in book.local_orders['B']  # Order I removed
    assert "J" in book.local_orders['B']  # Order J still present

    # Trade 12: Sell 5 units at 100 - should fill order J
    trade12 = make_trade(price=100, size=4, side="A")
    fills12 = book.on_trade(trade12)
    assert [(lo.order.client_oid, size) for lo, size in fills12] == [("J", 2)]  # Order J gets filled
    assert book.price_to_level['B'][100].size == 1
    assert "J" not in book.local_orders['B']  # Order J removed

    # === FINAL VERIFICATION ===

    # Verify all orders have been properly removed
    assert len(book.local_orders['B']) == 0
    assert len(book.local_orders['A']) == 0

    # Verify market volume has been consumed
    assert book.price_to_level['B'][100].size == 1
    assert book.price_to_level['B'][101].size == 0
    assert book.price_to_level['B'][99].size == 0
    assert book.price_to_level['B'][98].size == 0
    assert book.price_to_level['A'][102].size == 0
    assert book.price_to_level['A'][103].size == 0

    # Verify price level deques are empty
    assert len(book.price_to_level['B'][100].local_orders) == 0
    assert len(book.price_to_level['B'][101].local_orders) == 0
    assert len(book.price_to_level['B'][99].local_orders) == 0
    assert len(book.price_to_level['B'][98].local_orders) == 0
    assert len(book.price_to_level['A'][102].local_orders) == 0
    assert len(book.price_to_level['A'][103].local_orders) == 0
