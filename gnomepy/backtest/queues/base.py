from abc import ABC, abstractmethod
from typing import Deque

from gnomepy.backtest.exchanges.mbp.types import LocalOrder


class QueueModel(ABC):
    """
    Abstract interface for queue modeling.
    """
    @abstractmethod
    def on_modify(self, previous_quantity: int, new_quantity: int, local_queue: Deque[LocalOrder]) -> None:
        """
        Distribute removed volume against each order's phantom.
        """
        ...

    def on_trade(self, trade_size: int, local_queue: Deque[LocalOrder]) -> list[tuple[LocalOrder, int]]:
        """
        Allocates trade across each order based on phantom and remaining.
        Returns list of (LocalOrder, fill_size).
        """
        phantom_volume_consumed = None
        for local_order in local_queue:
            if phantom_volume_consumed is None:
                # We only care about how much volume the first order consumed
                phantom_volume_consumed = max(0, local_order.phantom_volume)

            local_order.phantom_volume -= trade_size
            local_order.cumulative_traded_quantity += trade_size

        filled_orders = []
        remaining_volume = trade_size - (0 if phantom_volume_consumed is None else phantom_volume_consumed)

        while local_queue and remaining_volume > 0:
            local_order = local_queue[0]
            if local_order.phantom_volume <= 0 and remaining_volume > 0:
                filled_qty = min(local_order.remaining, remaining_volume)
                remaining_volume -= filled_qty
                local_order.remaining -= filled_qty
                filled_orders.append((local_order, filled_qty))

                if local_order.remaining == 0:
                    local_queue.popleft()
            else:
                break

        return filled_orders
