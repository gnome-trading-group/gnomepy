"""
Volatility model abstraction for circuit breaker decisions.

Provides an ABC for predicting forward absolute price movement in bps.
"""

from abc import ABC, abstractmethod

import numpy as np


class VolatilityModel(ABC):
    """Abstract base class for volatility models.

    Used both as a circuit breaker (stop quoting when predicted movement
    exceeds a threshold) and to supply the sigma parameter to the AS
    spread/reservation price formulas.
    """

    @abstractmethod
    def predict(self, listing_data: dict[str, np.ndarray]) -> float | None:
        """Predict per-tick volatility in bps.

        Parameters
        ----------
        listing_data : dict[str, np.ndarray]
            Dict mapping column names to numpy arrays of historical values.

        Returns
        -------
        float or None
            Predicted per-tick volatility in bps, or None if insufficient data.
        """

    @property
    @abstractmethod
    def min_lookback(self) -> int:
        """Minimum ticks needed before prediction is possible."""

    @property
    @abstractmethod
    def horizon(self) -> int:
        """Forward horizon (in ticks) used for label estimation.

        Longer horizons produce smoother volatility estimates.  All models
        return per-tick bps directly, so the signal converts via
        ``sigma_per_tick = predicted_bps / 1e4``.
        """
