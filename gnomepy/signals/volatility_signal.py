from __future__ import annotations

from abc import abstractmethod

from gnomepy.java.schemas import JavaSchema
from gnomepy.signals.registrable import Registrable


class VolatilitySignal(Registrable):
    """Base class for volatility estimates measured in basis points.

    A 50 bps volatility means "expected price variation of ~50 bps
    over the model's horizon." Used to size spreads, set risk limits, etc.

    Subclass and implement update() and get_prediction().
    """

    @abstractmethod
    def update(self, timestamp: int, data: JavaSchema):
        """Feed market data for the current tick."""
        ...

    @abstractmethod
    def get_prediction(self) -> float:
        """Current volatility estimate in basis points."""
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """False during warmup period."""
        ...

    @abstractmethod
    def reset(self):
        """Clear internal state."""
        ...
