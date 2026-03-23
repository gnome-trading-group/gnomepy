from __future__ import annotations

from abc import ABC, abstractmethod

from gnomepy.java.schemas import JavaSchema


class BpsSignal(ABC):
    """Base class for directional signals measured in basis points.

    A +10 bps signal means "price will move up ~10 bps from current level."
    Negative values indicate bearish predictions.

    Subclass and implement update() and get_prediction().
    """

    @abstractmethod
    def update(self, timestamp: int, data: JavaSchema):
        """Feed market data for the current tick."""
        ...

    @abstractmethod
    def get_prediction(self) -> float:
        """Current signal in basis points."""
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """False during warmup period."""
        ...

    @abstractmethod
    def reset(self):
        """Clear internal state."""
        ...
