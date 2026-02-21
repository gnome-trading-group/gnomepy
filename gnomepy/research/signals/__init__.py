from abc import ABC, abstractmethod
from gnomepy.data.types import SchemaBase
from gnomepy.research.types import Intent


class Signal(ABC):

    def __init__(self):
        self._id = id(self)

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        return isinstance(other, Signal) and self._id == other._id

    @abstractmethod
    def process_new_tick(self, data: SchemaBase) -> list[Intent]:
        """
        Process market data and return intents.

        Returns:
            list of Intent objects
        """
        raise NotImplementedError


class PositionAwareSignal(Signal):

    @abstractmethod
    def process_new_tick(self, data: SchemaBase, positions: dict[int, float] = None) -> list[Intent]:
        """
        Process market data and return intents, considering current positions.

        Args:
            data: Market data to process
            positions: Dictionary mapping listing_id to current position size

        Returns:
            list of Intent objects
        """
        raise NotImplementedError
