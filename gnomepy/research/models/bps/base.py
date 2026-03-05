"""Abstract base class for bps return prediction models."""
from abc import ABC, abstractmethod
import numpy as np


class BpsModel(ABC):
    """Predict expected forward return in signed bps.

    Returns a signed float (positive = up, negative = down), or None
    when insufficient data is available.
    """

    @abstractmethod
    def predict(self, listing_data: dict[str, np.ndarray]) -> float | None:
        """Predict expected forward return in bps.

        Returns
        -------
        float or None
            Signed bps (e.g. +8.5 or -3.2), or None if insufficient data.
        """

    @property
    @abstractmethod
    def min_lookback(self) -> int:
        """Minimum ticks needed before prediction is possible."""
