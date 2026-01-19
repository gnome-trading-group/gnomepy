from dataclasses import dataclass

from gnomepy.registry.types import Listing


@dataclass
class Intent:
    listing: Listing
    side: str  # "buy" or "sell"
    confidence: float
    flatten: bool = False
    price: float | None = None  # Optional price for limit orders

@dataclass
class BasketIntent:
    intents: list[Intent]
    proportions: list[float]
    flatten: bool = False