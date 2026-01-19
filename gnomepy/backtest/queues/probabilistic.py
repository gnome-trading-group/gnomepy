from typing import Deque

from gnomepy.backtest.exchanges.mbp.types import LocalOrder
from gnomepy.backtest.queues.base import QueueModel


class ProbabilisticQueueModel(QueueModel):
    """
    Probabilistic queue model with a configurable probability that cancels
    occur ahead of our order. Depth increases are assumed to arrive behind us.

    Let p \in [0, 1] be the probability cancels occur ahead. When displayed
    quantity decreases by D, we reduce phantom by p * D (floored at zero).

    This model interpolates between optimistic (p=1) and fully pessimistic
    risk-averse (effectively clamping at new depth). Note: with p<1 there
    can remain positive phantom after a decrease reflecting expected cancels
    happening partially behind us.
    """

    def __init__(self, cancel_ahead_probability: float = 0.5) -> None:
        if cancel_ahead_probability < 0 or cancel_ahead_probability > 1:
            raise ValueError("cancel_ahead_probability must be in [0, 1]")
        self.cancel_ahead_probability = cancel_ahead_probability

    def on_modify(self, previous_quantity: int, new_quantity: int, local_queue: Deque[LocalOrder]) -> None:
        if not local_queue:
            return

        removed_volume = previous_quantity - new_quantity
        if removed_volume <= 0:
            # Depth increased or unchanged: assume additions are behind us
            return

        # Expected volume canceled ahead of us
        expected_ahead = int(round(self.cancel_ahead_probability * removed_volume))
        if expected_ahead <= 0:
            return

        for local_order in local_queue:
            local_order.phantom_volume = max(0, local_order.phantom_volume - expected_ahead)


