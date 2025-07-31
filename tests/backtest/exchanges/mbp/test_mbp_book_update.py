import pytest
from gnomepy.backtest.exchanges.mbp.mbp_book import MBPBook, OrderBookLevel
from gnomepy.data.types import BidAskPair
from tests.backtest.exchanges.mbp.test_mbp_book import DummyQueueModel, make_order

def make_bid_ask_pair(bid_px, bid_sz, ask_px, ask_sz):
    """Helper function to create BidAskPair objects for testing."""
    return BidAskPair(
        bid_px=bid_px,
        bid_sz=bid_sz,
        ask_px=ask_px,
        ask_sz=ask_sz,
        bid_ct=1,
        ask_ct=1,
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
    "market_bids, market_asks, local_orders, market_update, expected_bid_sizes, expected_ask_sizes, expect_error",
    [
        # Basic market level updates
        (
            [(100, 10)],
            [(102, 10)],
            [],  # No local orders
            [make_bid_ask_pair(100, 15, 102, 12)],
            {100: 15},  # Bid size updated
            {102: 12},  # Ask size updated
            False,
        ),
        # Multiple price levels updated
        (
            [(100, 10), (99, 5)],
            [(102, 10), (103, 8)],
            [],  # No local orders
            [
                make_bid_ask_pair(100, 15, 102, 12),
                make_bid_ask_pair(99, 7, 103, 6)
            ],
            {100: 15, 99: 7},  # Both bid sizes updated
            {102: 12, 103: 6},  # Both ask sizes updated
            False,
        ),
        # New price level added
        (
            [(100, 10)],
            [(102, 10)],
            [],  # No local orders
            [
                make_bid_ask_pair(100, 10, 101, 5),
                make_bid_ask_pair(99, 5, 102, 10),
            ],
            {99: 5, 100: 10},
            {102: 10, 101: 5},
            False,
        ),
        # Price level removed (size=0, no local orders)
        (
            [(100, 10)],
            [(102, 10)],
            [],  # No local orders
            [make_bid_ask_pair(100, 0, 102, 10)],  # Bid level size=0, no local orders
            {},  # Bid level removed
            {102: 10},  # Ask size unchanged
            False,
        ),
        # Multiple levels removed (size=0, no local orders)
        (
            [(100, 10), (99, 5)],
            [(102, 10), (103, 8)],
            [],  # No local orders
            [make_bid_ask_pair(99, 5, 103, 8)],
            {99: 5},  # 100 removed, 99 unchanged
            {103: 8},  # 102 removed, 103 unchanged
            False,
        ),
        # Zero trade size - levels removed (no local orders)
        (
            [(100, 10)],
            [(102, 10)],
            [],  # No local orders
            [make_bid_ask_pair(100, 0, 102, 0)],
            {},  # Bid level removed
            {},  # Ask level removed
            False,
        ),
        # Empty market update
        (
            [(100, 10)],
            [(102, 10)],
            [],  # No local orders
            [],
            {},
            {},
            False,
        ),
        # Edge case - very large sizes
        (
            [(100, 10)],
            [(102, 10)],
            [],  # No local orders
            [make_bid_ask_pair(100, 1000000, 102, 999999)],
            {100: 1000000},  # Very large bid size
            {102: 999999},  # Very large ask size
            False,
        ),
        # Edge case - very small sizes
        (
            [(100, 10)],
            [(102, 10)],
            [],  # No local orders
            [make_bid_ask_pair(100, 1, 102, 1)],
            {100: 1},  # Very small bid size
            {102: 1},  # Very small ask size
            False,
        ),
        # Price level with no change
        (
            [(100, 10)],
            [(102, 10)],
            [],  # No local orders
            [make_bid_ask_pair(100, 10, 102, 10)],  # Same sizes
            {100: 10},  # Bid size unchanged
            {102: 10},  # Ask size unchanged
            False,
        ),
        # Price level maintained (size=0 but has local orders)
        (
            [(100, 10)],
            [(102, 10)],
            [make_order(price=100, size=5, side="B", client_oid="A")],  # Local order at 100
            [make_bid_ask_pair(100, 0, 102, 10)],  # Bid level size=0 but has local orders
            {100: 0},  # Bid level maintained with size=0 (has local orders)
            {102: 10},  # Ask size unchanged
            False,
        ),
        (
            [(99, 11)],
            [(103, 11)],
            [
                make_order(price=99, size=5, side="B", client_oid="A"),
                make_order(price=103, size=5, side="A", client_oid="A"),
            ],
            [make_bid_ask_pair(100, 10, 102, 10)],
            {99: 0, 100: 10},
            {102: 10, 103: 0},
            False,
        ),
        (
            [(99, 11)],
            [(103, 11)],
            [
                make_order(price=99, size=5, side="B", client_oid="A"),
                make_order(price=103, size=5, side="A", client_oid="A"),
            ],
            [make_bid_ask_pair(103, 10, 104, 10)],
            {99: 0, 103: 10},
            {103: 10, 104: 10},
            True,
        ),
    ]
)
def test_on_market_update_price_levels(
    market_bids,
    market_asks,
    local_orders,
    market_update,
    expected_bid_sizes,
    expected_ask_sizes,
    expect_error,
):
    book = setup_book_with_local_orders_and_market_volume(
        local_orders=local_orders,
        market_bids=market_bids,
        market_asks=market_asks,
    )
    
    if expect_error:
        with pytest.raises(ValueError):
            book.on_market_update(market_update)
    else:
        book.on_market_update(market_update)
        
        # Check bid side sizes
        for price, expected_size in expected_bid_sizes.items():
            assert book.price_to_level['B'][price].size == expected_size
        
        # Check ask side sizes
        for price, expected_size in expected_ask_sizes.items():
            assert book.price_to_level['A'][price].size == expected_size
        
        # Verify all expected levels are in price lists
        for price in expected_bid_sizes:
            assert price in book.bids
        
        for price in expected_ask_sizes:
            assert price in book.asks

def test_on_market_update_price_level_management():
    """Test that price levels are properly added/removed during market updates."""
    book = MBPBook(queue_model=DummyQueueModel())
    
    # Add market volume
    level = OrderBookLevel(price=100, size=10)
    book.price_to_level['B'][100] = level
    book.bids.append(100)
    
    # Market update that sets size=0 (no local orders, so level is removed)
    market_update = [make_bid_ask_pair(100, 0, 102, 10)]
    book.on_market_update(market_update)
    
    # Verify price level is removed from bids list (size=0, no local orders)
    assert 100 not in book.bids
    assert book.price_to_level['B'].get(100) is None
    
    # Market update that maintains existing level (not in update)
    market_update2 = [make_bid_ask_pair(99, 5, 102, 10)]  # 100 not in update
    book.on_market_update(market_update2)
    
    # Verify 100 is still not in bids (was already removed)
    assert 100 not in book.bids
    assert 99 in book.bids
    assert book.price_to_level['B'][99].size == 5

def test_on_market_update_empty_book():
    """Test on_market_update with completely empty book."""
    book = MBPBook(queue_model=DummyQueueModel())
    market_update = [make_bid_ask_pair(100, 10, 102, 10)]
    
    book.on_market_update(market_update)
    
    # Verify book state is updated
    assert book.price_to_level['B'][100].size == 10
    assert book.price_to_level['A'][102].size == 10
    assert 100 in book.bids
    assert 102 in book.asks

def test_on_market_update_no_local_orders():
    """Test on_market_update with market volume but no local orders."""
    book = MBPBook(queue_model=DummyQueueModel())
    
    # Add market volume without local orders
    level = OrderBookLevel(price=100, size=10)
    book.price_to_level['B'][100] = level
    book.bids.append(100)
    
    market_update = [make_bid_ask_pair(100, 15, 102, 10)]
    book.on_market_update(market_update)
    
    # Verify market volume is updated
    assert book.price_to_level['B'][100].size == 15
    assert book.price_to_level['A'][102].size == 10
    assert len(book.local_orders['B']) == 0
    assert len(book.local_orders['A']) == 0

def test_on_market_update_price_list_maintenance():
    """Test that price lists (bids/asks) are properly maintained during updates."""
    book = MBPBook(queue_model=DummyQueueModel())
    
    for price, size in [(100, 10), (99, 5), (98, 3)]:
        level = OrderBookLevel(price=price, size=size)
        book.price_to_level['B'][price] = level
        book.bids.append(price)
    
    for price, size in [(102, 10), (103, 8), (104, 6)]:
        level = OrderBookLevel(price=price, size=size)
        book.price_to_level['A'][price] = level
        book.asks.append(price)
    
    # Verify initial state
    assert book.bids == [100, 99, 98]  # Descending order
    assert book.asks == [102, 103, 104]  # Ascending order
    
    # Market update that removes some levels and adds new ones
    market_update = [
        make_bid_ask_pair(99, 5, 103, 8),
        make_bid_ask_pair(98, 3, 104, 6),
        make_bid_ask_pair(97, 7, 105, 4),   # Add new levels
    ]
    
    book.on_market_update(market_update)
    
    # Verify price lists are updated correctly
    assert book.bids == [99, 98, 97]  # 100 removed (size=0, no local orders), 97 added
    assert book.asks == [103, 104, 105]  # 102 removed (size=0, no local orders), 105 added
    
    # Verify sizes are correct
    assert book.price_to_level['B'][99].size == 5
    assert book.price_to_level['B'][98].size == 3
    assert book.price_to_level['B'][97].size == 7
    assert book.price_to_level['A'][103].size == 8
    assert book.price_to_level['A'][104].size == 6
    assert book.price_to_level['A'][105].size == 4 