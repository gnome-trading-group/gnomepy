from __future__ import annotations

from abc import abstractmethod

from gnomepy.java.schemas import JavaSchema
from gnomepy.signals.registrable import Registrable


class FairValueModel(Registrable):
    """Base class for fair value estimates.

    Returns a fair value price in raw units (same scale as bid/ask from the book).
    The market maker quotes around this value instead of raw mid.
    """

    @abstractmethod
    def update(self, timestamp: int, data: JavaSchema) -> None:
        """Feed market data for the current tick."""
        ...

    @abstractmethod
    def get_fair_value(self) -> int:
        """Current fair value estimate in raw price units."""
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """False during warmup period."""
        ...
