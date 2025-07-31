from typing import Deque

from gnomepy.backtest.exchanges.mbp.types import LocalOrder
from gnomepy.backtest.queues.base import QueueModel


class RiskAverseQueueModel(QueueModel):
    """
    Risk-averse queue model represents a very sad queue model.

    Every cancel is assumed to happen behind the order's position, and every trade
    happens in front of the order. A fill is only possible once all volume is cleared.

    Here's an example:

    T0 Queue (initial state):
    [Order(500), Order(100)]

    T1 Queue (local order comes in):
    [Order(500), Order(100), LocalOrder(50)]

    T2 Queue (another two orders come in):
    [Order(500), Order(100), LocalOrder(50), Order(200), Order(200)]

    T3 Queue (trade for 200 lots)
    [Order(300), Order(100), LocalOrder(50), Order(200), Order(200)]

    T4 Queue (a cancel order on the first two orders (300, 100) is assumed to be behind us)
    Actual queue: [LocalOrder(50), Order(200), Order(200)]
    Our queue model: [Order(300), Order(100), LocalOrder(50)]
    """

    def on_modify(self, previous_quantity: int, new_quantity: int, local_queue: Deque[LocalOrder]) -> None:
        for local_order in local_queue:
            local_order.phantom_volume = min(local_order.phantom_volume, new_quantity)
