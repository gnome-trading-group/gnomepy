import pytest
from gnomepy.backtest.exchanges.mbp.mbp_book import MBPBook, OrderBookLevel
from tests.backtest.exchanges.mbp.test_mbp_book import DummyQueueModel, make_order, make_local_order

@pytest.mark.parametrize(
    "orders_to_add, cancel_client_oid, expected_result, expected_remaining_orders",
    [
        # Cancel existing order
        (
            [make_order(price=100, size=5, side="B", client_oid="A")],
            "A",
            True,
            [],
        ),
        # Cancel non-existent order
        (
            [make_order(price=100, size=5, side="B", client_oid="A")],
            "B",
            False,
            ["A"],
        ),
        # Cancel one of multiple orders
        (
            [
                make_order(price=100, size=5, side="B", client_oid="A"),
                make_order(price=101, size=3, side="B", client_oid="B"),
            ],
            "A",
            True,
            ["B"],
        ),
        # Cancel ask side order
        (
            [make_order(price=102, size=5, side="A", client_oid="C")],
            "C",
            True,
            [],
        ),
        # Cancel from mixed bid/ask orders
        (
            [
                make_order(price=100, size=5, side="B", client_oid="A"),
                make_order(price=102, size=3, side="A", client_oid="B"),
            ],
            "B",
            True,
            ["A"],
        ),
        # Cancel from empty book
        (
            [],
            "A",
            False,
            [],
        ),
        # Cancel order with different price levels
        (
            [
                make_order(price=100, size=5, side="B", client_oid="A"),
                make_order(price=99, size=3, side="B", client_oid="B"),
                make_order(price=101, size=2, side="B", client_oid="C"),
            ],
            "B",
            True,
            ["A", "C"],
        ),
        # Cancel order with same price level
        (
            [
                make_order(price=100, size=5, side="B", client_oid="A"),
                make_order(price=100, size=3, side="B", client_oid="B"),
            ],
            "A",
            True,
            ["B"],
        ),
    ]
)
def test_cancel_order(
    orders_to_add,
    cancel_client_oid,
    expected_result,
    expected_remaining_orders,
):
    book = MBPBook(queue_model=DummyQueueModel())
    
    for order in orders_to_add:
        book.add_local_order(order)
    
    result = book.cancel_order(cancel_client_oid)
    
    assert result is expected_result
    
    all_remaining = []
    for side in ['B', 'A']:
        all_remaining.extend(book.local_orders[side].keys())
    
    assert sorted(all_remaining) == sorted(expected_remaining_orders) 