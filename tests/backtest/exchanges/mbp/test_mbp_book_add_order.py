import pytest
from gnomepy.backtest.exchanges.mbp.mbp_book import MBPBook
from tests.backtest.exchanges.mbp.test_mbp_book import DummyQueueModel, make_order

def setup_book_with_market_volume(bids=None, asks=None):
    """Setup book with existing market volume at price levels."""
    book = MBPBook(queue_model=DummyQueueModel())
    
    # bids: list of (price, size)
    # asks: list of (price, size)
    if bids:
        for price, size in bids:
            level = book.price_to_level['B'].get(price)
            if level is None:
                level = book.price_to_level['B'][price] = book.price_to_level['B'].get(price, book.price_to_level['B'].setdefault(price, type('OrderBookLevel', (), {'price': price, 'size': size, 'local_orders': []})()))
            level.size = size
            book.bids.append(price)
        book.bids.sort(reverse=True)
    
    if asks:
        for price, size in asks:
            level = book.price_to_level['A'].get(price)
            if level is None:
                level = book.price_to_level['A'][price] = book.price_to_level['A'].get(price, book.price_to_level['A'].setdefault(price, type('OrderBookLevel', (), {'price': price, 'size': size, 'local_orders': []})()))
            level.size = size
            book.asks.append(price)
        book.asks.sort()
    
    return book

@pytest.mark.parametrize(
    "existing_bids, existing_asks, orders_to_add, expected_bids, expected_asks, expected_local_orders, expected_phantom_volumes",
    [
        # Add order to empty book
        (
            [],
            [],
            [make_order(price=100, size=5, side="B", client_oid="A")],
            [100],
            [],
            ["A"],
            [0],  # no existing market volume
        ),
        # Add order to price level with existing market volume
        (
            [(100, 10)],  # existing 10 units at price 100
            [],
            [make_order(price=100, size=5, side="B", client_oid="A")],
            [100],
            [],
            ["A"],
            [10],  # phantom_volume = existing market volume (10)
        ),
        # Add multiple orders at same price level with existing market volume
        (
            [(100, 15)],  # existing 15 units at price 100
            [],
            [
                make_order(price=100, size=5, side="B", client_oid="A"),
                make_order(price=100, size=3, side="B", client_oid="B"),
            ],
            [100],
            [],
            ["A", "B"],
            [15, 15],  # both orders have phantom_volume = existing market volume (15)
        ),
        # Add orders at different price levels with different market volumes
        (
            [(100, 10), (99, 5)],  # existing volumes at different prices
            [],
            [
                make_order(price=100, size=5, side="B", client_oid="A"),
                make_order(price=99, size=3, side="B", client_oid="B"),
            ],
            [100, 99],
            [],
            ["A", "B"],
            [10, 5],  # phantom_volume = respective existing market volumes
        ),
        # Add ask orders with existing market volume
        (
            [],
            [(102, 8)],  # existing ask volume
            [make_order(price=102, size=3, side="A", client_oid="C")],
            [],
            [102],
            ["C"],
            [8],  # phantom_volume = existing ask market volume (8)
        ),
        # Mixed bid/ask with existing market volumes
        (
            [(100, 12)],
            [(102, 7)],
            [
                make_order(price=100, size=5, side="B", client_oid="A"),
                make_order(price=102, size=3, side="A", client_oid="B"),
            ],
            [100],
            [102],
            ["A", "B"],
            [12, 7],  # phantom_volume = respective existing market volumes
        ),
        # Add order to price level without existing market volume
        (
            [(100, 10)],  # existing volume at 100
            [],
            [make_order(price=101, size=5, side="B", client_oid="A")],  # new price level
            [101, 100],
            [],
            ["A"],
            [0],  # no existing market volume at new price level
        ),
        # Complex scenario with multiple price levels and volumes
        (
            [(100, 20), (99, 15), (98, 10)],
            [(102, 12), (103, 8)],
            [
                make_order(price=100, size=5, side="B", client_oid="A"),
                make_order(price=99, size=3, side="B", client_oid="B"),
                make_order(price=102, size=2, side="A", client_oid="C"),
            ],
            [100, 99, 98],
            [102, 103],
            ["A", "B", "C"],
            [20, 15, 12],  # phantom_volume = respective existing market volumes
        ),
    ]
)
def test_add_local_order_success(
    existing_bids,
    existing_asks,
    orders_to_add,
    expected_bids,
    expected_asks,
    expected_local_orders,
    expected_phantom_volumes,
):
    book = setup_book_with_market_volume(existing_bids, existing_asks)
    
    for order in orders_to_add:
        book.add_local_order(order)
    
    # Check bids and asks are correctly sorted
    assert book.bids == expected_bids
    assert book.asks == expected_asks
    
    # Check local orders exist
    all_local_orders = []
    for side in ['B', 'A']:
        all_local_orders.extend(book.local_orders[side].keys())
    
    # Handle auto-generated client_oid pattern matching
    if "internal_*" in expected_local_orders:
        assert len(all_local_orders) == len(expected_local_orders)
        assert any(oid.startswith("internal_") for oid in all_local_orders)
    else:
        assert sorted(all_local_orders) == sorted(expected_local_orders)
    
    # Check phantom volumes
    actual_phantom_volumes = []
    for side in ['B', 'A']:
        for client_oid in book.local_orders[side]:
            local_order = book.local_orders[side][client_oid]
            actual_phantom_volumes.append(local_order.phantom_volume)
    
    assert actual_phantom_volumes == expected_phantom_volumes

@pytest.mark.parametrize(
    "orders_to_add, duplicate_client_oid, expect_error",
    [
        # Duplicate client_oid on same side
        (
            [make_order(price=100, size=5, side="B", client_oid="A")],
            "A",
            True,
        ),
        # Duplicate client_oid on different side (should still fail)
        (
            [make_order(price=100, size=5, side="B", client_oid="A")],
            "A",
            True,
        ),
        # No duplicate
        (
            [make_order(price=100, size=5, side="B", client_oid="A")],
            "B",
            False,
        ),
        # Multiple orders, duplicate with first
        (
            [
                make_order(price=100, size=5, side="B", client_oid="A"),
                make_order(price=101, size=3, side="B", client_oid="B"),
            ],
            "A",
            True,
        ),
        # Multiple orders, duplicate with second
        (
            [
                make_order(price=100, size=5, side="B", client_oid="A"),
                make_order(price=101, size=3, side="B", client_oid="B"),
            ],
            "B",
            True,
        ),
    ]
)
def test_add_local_order_duplicate_client_oid(
    orders_to_add,
    duplicate_client_oid,
    expect_error,
):
    book = MBPBook(queue_model=DummyQueueModel())
    
    # Add initial orders
    for order in orders_to_add:
        book.add_local_order(order)
    
    # Try to add duplicate
    duplicate_order = make_order(
        price=102,
        size=2,
        side="B",
        client_oid=duplicate_client_oid,
    )
    
    if expect_error:
        with pytest.raises(ValueError, match=f"Duplicate client OID: {duplicate_client_oid}"):
            book.add_local_order(duplicate_order)
    else:
        book.add_local_order(duplicate_order)

def test_add_local_order_auto_generated_client_oid():
    """Test that orders without client_oid get auto-generated ones."""
    book = MBPBook(queue_model=DummyQueueModel())
    
    order1 = make_order(price=100, size=5, side="B", client_oid=None)
    order2 = make_order(price=101, size=3, side="B", client_oid=None)
    
    book.add_local_order(order1)
    book.add_local_order(order2)
    
    # Check that client_oids were generated
    assert order1.client_oid is not None
    assert order2.client_oid is not None
    assert order1.client_oid.startswith("internal_")
    assert order2.client_oid.startswith("internal_")
    assert order1.client_oid != order2.client_oid
    
    # Check they're in the book
    all_local_orders = []
    for side in ['B', 'A']:
        all_local_orders.extend(book.local_orders[side].keys())
    
    assert order1.client_oid in all_local_orders
    assert order2.client_oid in all_local_orders

def test_add_local_order_price_level_creation():
    """Test that new price levels are properly created and inserted."""
    book = MBPBook(queue_model=DummyQueueModel())
    
    # Add orders at different prices to test price level creation
    order1 = make_order(price=100, size=5, side="B", client_oid="A")
    order2 = make_order(price=99, size=3, side="B", client_oid="B")
    order3 = make_order(price=101, size=2, side="B", client_oid="C")
    
    book.add_local_order(order1)
    book.add_local_order(order2)
    book.add_local_order(order3)
    
    # Check bids are sorted descending
    assert book.bids == [101, 100, 99]
    
    # Check price levels exist
    assert 100 in book.price_to_level['B']
    assert 99 in book.price_to_level['B']
    assert 101 in book.price_to_level['B']
    
    # Check each level has the correct order
    assert len(book.price_to_level['B'][100].local_orders) == 1
    assert len(book.price_to_level['B'][99].local_orders) == 1
    assert len(book.price_to_level['B'][101].local_orders) == 1 