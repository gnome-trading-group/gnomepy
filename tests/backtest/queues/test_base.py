import pytest
from collections import deque
from gnomepy.backtest.queues.base import QueueModel
from gnomepy.backtest.exchanges.mbp.types import LocalOrder
from gnomepy.data.types import Order, OrderType, TimeInForce

class DummyQueueModel(QueueModel):
    def on_modify(self, previous_quantity, new_quantity, local_queue):
        pass  # Not needed for on_trade tests

def make_order(
    price=10000, size=10, side="B", order_type=OrderType.LIMIT, tif=TimeInForce.GTC,
    exchange_id=1, security_id=1, client_oid=None
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
    remaining, phantom_volume=0, cumulative_traded_quantity=0, **order_kwargs
):
    return LocalOrder(
        order=make_order(**order_kwargs),
        remaining=remaining,
        phantom_volume=phantom_volume,
        cumulative_traded_quantity=cumulative_traded_quantity,
    )

@pytest.mark.parametrize(
    "local_orders, trade_size, expected_fills, expected_state, expected_deque_client_oids",
    [
        # No orders
        ([], 5, [], [], []),
        # Trade smaller than phantom
        (
                [dict(remaining=10, phantom_volume=7, client_oid="A")],
                5,
                [],
                [dict(phantom_volume=2, cumulative_traded_quantity=5, remaining=10)],
                ["A"]
        ),
        # Trade exactly phantom
        (
                [dict(remaining=10, phantom_volume=5, client_oid="A")],
                5,
                [],
                [dict(phantom_volume=0, cumulative_traded_quantity=5, remaining=10)],
                ["A"]
        ),
        # Trade consumes phantom and partial fill
        (
                [dict(remaining=10, phantom_volume=3, client_oid="A")],
                5,
                [("A", 2)],
                [dict(phantom_volume=-2, cumulative_traded_quantity=5, remaining=8)],
                ["A"]
        ),
        # Trade consumes phantom and fills entire order (should be removed)
        (
                [dict(remaining=2, phantom_volume=1, client_oid="A")],
                5,
                [("A", 2)],
                [dict(phantom_volume=-4, cumulative_traded_quantity=5, remaining=0)],
                []
        ),
        # Trade fills multiple orders, last order partially filled
        (
                [
                    dict(remaining=2, phantom_volume=1, client_oid="A"),
                    dict(remaining=3, phantom_volume=1, client_oid="B"),
                    dict(remaining=4, phantom_volume=1, client_oid="C")
                ],
                7,
                [("A", 2), ("B", 3), ("C", 1)],
                [dict(remaining=0), dict(remaining=0), dict(remaining=3, phantom_volume=-6)],
                ["C"]
        ),
        # Trade larger than all orders (all should be removed)
        (
                [
                    dict(remaining=2, phantom_volume=1, client_oid="A"),
                    dict(remaining=3, phantom_volume=0, client_oid="B")
                ],
                10,
                [("A", 2), ("B", 3)],
                [dict(remaining=0), dict(remaining=0)],
                []
        ),
        # Trade with zero phantom, partial fill
        (
                [dict(remaining=5, phantom_volume=0, client_oid="A")],
                3,
                [("A", 3)],
                [dict(remaining=2)],
                ["A"]
        ),
        # Trade with zero phantom, full fill
        (
                [dict(remaining=3, phantom_volume=0, client_oid="A")],
                3,
                [("A", 3)],
                [dict(remaining=0)],
                []
        ),
        # Trade with negative phantom, partial fill
        (
                [dict(remaining=5, phantom_volume=-2, client_oid="A")],
                3,
                [("A", 3)],
                [dict(remaining=2)],
                ["A"]
        ),
        # Trade with negative phantom, full fill
        (
                [dict(remaining=3, phantom_volume=-2, client_oid="A")],
                3,
                [("A", 3)],
                [dict(remaining=0)],
                []
        ),
        # Trade with three orders, all removed
        (
                [
                    dict(remaining=2, phantom_volume=1, client_oid="A"),
                    dict(remaining=3, phantom_volume=0, client_oid="B"),
                    dict(remaining=4, phantom_volume=0, client_oid="C")
                ],
                20,
                [("A", 2), ("B", 3), ("C", 4)],
                [dict(remaining=0), dict(remaining=0), dict(remaining=0)],
                []
        ),
        # Trade with three orders, one filled
        (
                [
                    dict(remaining=2, phantom_volume=1, client_oid="A"),
                    dict(remaining=3, phantom_volume=5, client_oid="B"),
                    dict(remaining=4, phantom_volume=5, client_oid="C")
                ],
                3,
                [("A", 2)],
                [dict(remaining=0), dict(remaining=3, phantom_volume=2), dict(remaining=4, phantom_volume=2)],
                ["B", "C"]
        ),
    ]
)
def test_on_trade(local_orders, trade_size, expected_fills, expected_state, expected_deque_client_oids):
    # Build LocalOrder objects
    los = [make_local_order(**kwargs) for kwargs in local_orders]
    dq = deque(los)
    model = DummyQueueModel()
    fills = model.on_trade(trade_size, dq)
    # Check fills
    assert [(lo.order.client_oid, qty) for lo, qty in fills] == expected_fills
    # Check state
    for lo, state in zip(los, expected_state):
        for k, v in state.items():
            assert getattr(lo, k) == v, f"{lo.order.client_oid}: {k} expected {v}, got {getattr(lo, k)}"
    # Check deque contents (client_oids of remaining orders)
    assert [lo.order.client_oid for lo in dq] == expected_deque_client_oids
