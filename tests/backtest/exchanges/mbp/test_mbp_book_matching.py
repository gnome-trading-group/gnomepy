import pytest
from gnomepy.backtest.exchanges.mbp.mbp_book import OrderMatch
from gnomepy.backtest.exchanges.mbp.types import LocalOrder
from tests.backtest.exchanges.mbp.test_mbp_book import make_order, setup_book

@pytest.mark.parametrize(
    "bids, asks, order, expected_matches",
    [
        # Simple buy order matches best ask
        (
            [],
            [(101, 5)],
            make_order(price=102, size=3, side="B"),
            [OrderMatch(price=101, size=3)],
        ),
        # Buy order matches multiple asks
        (
            [],
            [(101, 2), (102, 4)],
            make_order(price=102, size=5, side="B"),
            [OrderMatch(price=101, size=2), OrderMatch(price=102, size=3)],
        ),
        # Buy order matches all asks, not enough liquidity
        (
            [],
            [(101, 2), (102, 4)],
            make_order(price=102, size=10, side="B"),
            [OrderMatch(price=101, size=2), OrderMatch(price=102, size=4)],
        ),
        # Buy order with price below best ask (no match)
        (
            [],
            [(101, 5)],
            make_order(price=100, size=3, side="B"),
            [],
        ),
        # Sell order matches best bid
        (
            [(99, 5)],
            [],
            make_order(price=98, size=3, side="A"),
            [OrderMatch(price=99, size=3)],
        ),
        # Sell order matches multiple bids
        (
            [(99, 2), (98, 4)],
            [],
            make_order(price=97, size=5, side="A"),
            [OrderMatch(price=99, size=2), OrderMatch(price=98, size=3)],
        ),
        # Sell order matches all bids, not enough liquidity
        (
            [(99, 2), (98, 4)],
            [],
            make_order(price=97, size=10, side="A"),
            [OrderMatch(price=99, size=2), OrderMatch(price=98, size=4)],
        ),
        # Sell order with price above best bid (no match)
        (
            [(99, 5)],
            [],
            make_order(price=100, size=3, side="A"),
            [],
        ),
        # Buy order, no asks at all
        (
            [],
            [],
            make_order(price=102, size=3, side="B"),
            [],
        ),
        # Sell order, no bids at all
        (
            [],
            [],
            make_order(price=98, size=3, side="A"),
            [],
        ),
        # Buy order, exact match on ask price
        (
            [],
            [(101, 5)],
            make_order(price=101, size=5, side="B"),
            [OrderMatch(price=101, size=5)],
        ),
        # Sell order, exact match on bid price
        (
            [(99, 5)],
            [],
            make_order(price=99, size=5, side="A"),
            [OrderMatch(price=99, size=5)],
        ),
        # Buy order, partial fill on best ask
        (
            [],
            [(101, 2)],
            make_order(price=101, size=1, side="B"),
            [OrderMatch(price=101, size=1)],
        ),
        # Sell order, partial fill on best bid
        (
            [(99, 2)],
            [],
            make_order(price=99, size=1, side="A"),
            [OrderMatch(price=99, size=1)],
        ),
    ]
)
def test_get_matching_orders(bids, asks, order, expected_matches):
    book = setup_book(bids, asks)
    matches = book.get_matching_orders(order)
    assert [(m.price, m.size) for m in matches] == [(m.price, m.size) for m in expected_matches]

@pytest.mark.parametrize(
    "bids, asks, order, local_bids, local_asks, expect_self_fill",
    [
        # Self-filling on bid side
        (
            [(99, 5)],
            [],
            make_order(price=98, size=3, side="A"),
            {99: [LocalOrder(order=make_order(price=99, size=5, side="B", client_oid="X"), remaining=5)]},
            None,
            True,
        ),
        # Self-filling on ask side
        (
            [],
            [(101, 5)],
            make_order(price=102, size=3, side="B"),
            None,
            {101: [LocalOrder(order=make_order(price=101, size=5, side="A", client_oid="Y"), remaining=5)]},
            True,
        ),
        # No self-filling if no local orders
        (
            [(99, 5)],
            [],
            make_order(price=98, size=3, side="A"),
            None,
            None,
            False,
        ),
        # No self-filling if local orders on other price
        (
            [(99, 5)],
            [],
            make_order(price=98, size=3, side="A"),
            {98: [LocalOrder(order=make_order(price=98, size=5, side="B", client_oid="Z"), remaining=5)]},
            None,
            False,
        ),
    ]
)
def test_get_matching_orders_self_filling(bids, asks, order, local_bids, local_asks, expect_self_fill):
    book = setup_book(bids, asks, bid_locals=local_bids, ask_locals=local_asks)
    if expect_self_fill:
        with pytest.raises(ValueError, match="Self filling triggered"):
            book.get_matching_orders(order)
    else:
        book.get_matching_orders(order) 