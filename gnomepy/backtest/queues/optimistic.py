from typing import Deque

from gnomepy.backtest.exchanges.mbp.types import LocalOrder
from gnomepy.backtest.queues.base import QueueModel


class OptimisticQueueModel(QueueModel):
    """
    Optimistic queue model assumes the best case for our order:

    - Cancels are assumed to occur ahead of our position, reducing phantom first.
    - New orders are assumed to arrive behind us, not increasing phantom.

    Therefore, when displayed depth at the price level decreases, we deduct the
    removed volume from each local order's phantom, floored at zero. When depth
    increases, we leave phantom unchanged.
    """

    def on_modify(self, previous_quantity: int, new_quantity: int, local_queue: Deque[LocalOrder]) -> None:
        if not local_queue:
            return

        removed_volume = previous_quantity - new_quantity
        if removed_volume <= 0:
            # Depth increased or unchanged: assume adds are behind us
            return

        for local_order in local_queue:
            local_order.phantom_volume = max(0, local_order.phantom_volume - removed_volume)


