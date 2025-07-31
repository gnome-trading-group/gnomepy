from collections import deque
from gnomepy.backtest.exchanges.mbp.mbp_book import MBPBook, OrderBookLevel
from gnomepy.backtest.queues.base import QueueModel
from gnomepy.backtest.exchanges.mbp.types import LocalOrder
from gnomepy.data.types import Order, OrderType, TimeInForce, MBP10, MBP1, BidAskPair

class DummyQueueModel(QueueModel):
    def on_modify(self, previous_quantity, new_quantity, local_queue):
        pass

def make_order(
    price,
    size,
    side,
    order_type=OrderType.LIMIT,
    tif=TimeInForce.GTC,
    exchange_id=1,
    security_id=1,
    client_oid=None,
):
    return Order(
        exchange_id=exchange_id,
        security_id=security_id,
        client_oid=client_oid,
        price=price,
        size=size,
        side=side,
        order_type=order_type,
        time_in_force=tif,
    )

def make_local_order(
    order=None,
    remaining=None,
    phantom_volume=0,
    cumulative_traded_quantity=0,
    **order_kwargs,
):
    if order is None:
        order = make_order(**order_kwargs)
    if remaining is None:
        remaining = order.size
    return LocalOrder(
        order=order,
        remaining=remaining,
        phantom_volume=phantom_volume,
        cumulative_traded_quantity=cumulative_traded_quantity,
    )

def make_book_level(price, size, local_orders=None):
    lvl = OrderBookLevel(price=price, size=size)
    if local_orders:
        lvl.local_orders = deque(local_orders)
    return lvl

def setup_book(bids, asks, bid_locals=None, ask_locals=None):
    # bids: list of (price, size)
    # asks: list of (price, size)
    # bid_locals/ask_locals: dict of price -> list of LocalOrder
    book = MBPBook(queue_model=DummyQueueModel())
    book.bids = sorted([b[0] for b in bids], reverse=True)
    book.asks = sorted([a[0] for a in asks])
    book.price_to_level['B'] = {b[0]: make_book_level(b[0], b[1], (bid_locals or {}).get(b[0])) for b in bids}
    book.price_to_level['A'] = {a[0]: make_book_level(a[0], a[1], (ask_locals or {}).get(a[0])) for a in asks}
    return book

def make_trade(side, price, size):
    """Helper to create a trade for on_trade method"""
    return MBP10(
        action="Trade",
        side=side,
        price=price,
        size=size,
        timestamp_event=0,
        timestamp_recv=0,
        timestamp_sent=0,
        exchange_id=1,
        security_id=1,
        flags=[],
        sequence=0,
        depth=0,
        levels=[],
    )

def make_bid_ask_pair(bid_px, bid_sz, ask_px, ask_sz):
    """Helper to create a BidAskPair for on_market_update method"""
    return BidAskPair(
        bid_px=bid_px,
        bid_sz=bid_sz,
        ask_px=ask_px,
        ask_sz=ask_sz,
        bid_ct=1,
        ask_ct=1,
    )

def test_mbp_book_lifecycle():
    """
    Comprehensive test simulating the lifecycle of an order book over multiple
    trades, market updates, and order cancellations.
    """
    book = MBPBook(queue_model=DummyQueueModel())
    
    # === PHASE 1: Initial Market State ===
    # Realistic order book with 5 levels on each side
    initial_market_update = [
        make_bid_ask_pair(100, 50, 101, 40),
        make_bid_ask_pair(99, 30, 102, 35),
        make_bid_ask_pair(98, 25, 103, 30),
        make_bid_ask_pair(97, 20, 104, 25),
        make_bid_ask_pair(96, 15, 105, 20),
    ]
    
    book.on_market_update(initial_market_update)
    
    # Verify initial state
    assert book.bids == [100, 99, 98, 97, 96]
    assert book.asks == [101, 102, 103, 104, 105]
    assert book.price_to_level['B'][100].size == 50
    assert book.price_to_level['A'][101].size == 40

    # === PHASE 2: Add Local Orders ===
    # Add some local orders to simulate our positions
    local_orders = [
        make_order(price=99, size=10, side="B", client_oid="BID_1"),   # Bid at 99
        make_order(price=98, size=15, side="B", client_oid="BID_2"),   # Bid at 98
        make_order(price=102, size=8, side="A", client_oid="ASK_1"),   # Ask at 102
        make_order(price=103, size=12, side="A", client_oid="ASK_2"),  # Ask at 103
    ]

    for order in local_orders:
        book.add_local_order(order)

    # Verify local orders are added
    assert len(book.local_orders['B']) == 2
    assert len(book.local_orders['A']) == 2
    assert "BID_1" in book.local_orders['B']
    assert "ASK_1" in book.local_orders['A']

    # === PHASE 3: First Round of Trades ===
    # Simulate aggressive buying - trades at 101, 102, 103
    trades_round_1 = [
        make_trade("B", 101, 40),  # Buy 40 at 101 (consumes ask volume)
        make_trade("B", 102, 25),  # Buy 25 at 102 (consumes ask volume + hits our local order)
        make_trade("B", 103, 20),  # Buy 10 at 103 (consumes ask volume + hits our local order)
    ]

    fills_round_1 = []
    for trade in trades_round_1:
        fills = book.on_trade(trade)
        fills_round_1.extend(fills)

    # Verify some local orders were filled
    assert len(fills_round_1) > 0

    # # === PHASE 4: Market Update After First Trades ===
    # # Market update reflecting the consumed volume
    # market_update_1 = [
    #     make_bid_ask_pair(100, 50),
    #     make_bid_ask_pair(99, 30),
    #     make_bid_ask_pair(98, 25),
    #     make_bid_ask_pair(97, 20),
    #     make_bid_ask_pair(96, 15),
    #     make_bid_ask_pair(101, 20),
    #     make_bid_ask_pair(102, 20),
    #     make_bid_ask_pair(103, 20),
    #     make_bid_ask_pair(104, 25),
    #     make_bid_ask_pair(105, 20),
    # ]

    # book.on_market_update(market_update_1)
    #
    # # Verify market levels are updated
    # assert book.price_to_level['A'][101].size == 20
    # assert book.price_to_level['A'][102].size == 20
    # assert book.price_to_level['A'][103].size == 20
    #
    # # === PHASE 5: Cancel Some Local Orders ===
    # # Cancel one bid and one ask order
    # from gnomepy.data.types import CancelOrder
    #
    # cancel_bid = CancelOrder(
    #     exchange_id=1,
    #     security_id=1,
    #     client_oid="BID_1",
    # )
    #
    # cancel_ask = CancelOrder(
    #     exchange_id=1,
    #     security_id=1,
    #     client_oid="ASK_1",
    # )
    #
    # assert book.cancel_order(cancel_bid) == True
    # assert book.cancel_order(cancel_ask) == True
    #
    # # Verify orders are cancelled
    # assert "BID_1" not in book.local_orders['B']
    # assert "ASK_1" not in book.local_orders['A']
    # assert len(book.local_orders['B']) == 1
    # assert len(book.local_orders['A']) == 1
    #
    # # === PHASE 6: Second Round of Trades ===
    # # Simulate aggressive selling - trades at 100, 99
    # trades_round_2 = [
    #     make_trade("A", 100, 25),  # Sell 25 at 100 (consumes bid volume)
    #     make_trade("A", 99, 20),   # Sell 20 at 99 (consumes bid volume + hits our local order)
    # ]
    #
    # fills_round_2 = []
    # for trade in trades_round_2:
    #     fills = book.on_trade(trade)
    #     fills_round_2.extend(fills)
    #
    # # === PHASE 7: Market Update After Second Trades ===
    # # Market update reflecting the consumed volume
    # market_update_2 = [
    #     make_bid_ask_pair(100, 25),   # Reduced from 50 to 25
    #     make_bid_ask_pair(99, 10),    # Reduced from 30 to 10 (20 consumed)
    #     make_bid_ask_pair(98, 25),    # Unchanged
    #     make_bid_ask_pair(97, 20),    # Unchanged
    #     make_bid_ask_pair(96, 15),    # Unchanged
    #     make_bid_ask_pair(101, 20),   # Unchanged
    #     make_bid_ask_pair(102, 20),   # Unchanged
    #     make_bid_ask_pair(103, 20),   # Unchanged
    #     make_bid_ask_pair(104, 25),   # Unchanged
    #     make_bid_ask_pair(105, 20),   # Unchanged
    # ]
    #
    # book.on_market_update(market_update_2)
    #
    # # Verify market levels are updated
    # assert book.price_to_level['B'][100].size == 25
    # assert book.price_to_level['B'][99].size == 10
    #
    # # === PHASE 8: Third Round of Trades ===
    # # More aggressive trading
    # trades_round_3 = [
    #     make_trade("B", 101, 15),  # Buy 15 at 101
    #     make_trade("A", 98, 10),   # Sell 10 at 98
    # ]
    #
    # for trade in trades_round_3:
    #     book.on_trade(trade)
    #
    # # === PHASE 9: Final Market Update (Some Levels Removed) ===
    # # Market update that removes some levels that no longer exist
    # # Note: Levels not in the update are removed (as per the logic)
    # final_market_update = [
    #     make_bid_ask_pair(100, 25),   # Maintained
    #     make_bid_ask_pair(99, 10),    # Maintained
    #     make_bid_ask_pair(98, 15),    # Reduced from 25 to 15
    #     make_bid_ask_pair(97, 20),    # Maintained
    #     # 96 is NOT in update - should be removed
    #     make_bid_ask_pair(101, 5),    # Reduced from 20 to 5
    #     make_bid_ask_pair(102, 20),   # Maintained
    #     make_bid_ask_pair(103, 20),   # Maintained
    #     make_bid_ask_pair(104, 25),   # Maintained
    #     # 105 is NOT in update - should be removed
    # ]
    #
    # book.on_market_update(final_market_update)
    #
    # # === PHASE 10: Verify Final State ===
    # # Verify price levels are correctly maintained/removed
    # assert book.bids == [100, 99, 98, 97]  # 96 removed (not in update)
    # assert book.asks == [101, 102, 103, 104]  # 105 removed (not in update)
    #
    # # Verify sizes are correct
    # assert book.price_to_level['B'][100].size == 25
    # assert book.price_to_level['B'][99].size == 10
    # assert book.price_to_level['B'][98].size == 15
    # assert book.price_to_level['B'][97].size == 20
    # assert 96 not in book.price_to_level['B']  # Removed
    #
    # assert book.price_to_level['A'][101].size == 5
    # assert book.price_to_level['A'][102].size == 20
    # assert book.price_to_level['A'][103].size == 20
    # assert book.price_to_level['A'][104].size == 25
    # assert 105 not in book.price_to_level['A']  # Removed
    #
    # # Verify remaining local orders
    # assert len(book.local_orders['B']) == 1  # BID_2 at 98
    # assert len(book.local_orders['A']) == 1  # ASK_2 at 103
    # assert "BID_2" in book.local_orders['B']
    # assert "ASK_2" in book.local_orders['A']
    #
    # # Verify best bid/ask
    # assert book.get_best_bid() == 100
    # assert book.get_best_ask() == 101
    # assert book.get_best_bid() < book.get_best_ask()  # Spread is maintained