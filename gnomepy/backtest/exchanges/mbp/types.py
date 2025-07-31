from dataclasses import dataclass

from gnomepy.data.types import Order


@dataclass
class LocalOrder:
    order: Order
    remaining: int

    # Represents the amount of theoretical volume ahead of this order in the queue
    phantom_volume: int = 0
    # Tracking the amount of quantity deducted from trades so depth changes do not double count
    cumulative_traded_quantity: int = 0
