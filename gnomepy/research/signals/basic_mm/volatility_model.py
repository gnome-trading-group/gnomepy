"""
Volatility model abstraction for circuit breaker decisions.

Provides an ABC for predicting forward absolute price movement in bps,
plus a simple realized volatility implementation.
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


class RealizedVolatilityModel(VolatilityModel):
    """Simple realized volatility model.

    Computes rolling standard deviation of log returns over a window
    and returns per-tick volatility in bps.

    Parameters
    ----------
    window : int
        Lookback window for rolling std of log returns.
    horizon : int
        Retained for interface compatibility (not used in prediction).
    """

    def __init__(self, window: int = 100, horizon: int = 20):
        self.window = window
        self._horizon = horizon

    @property
    def min_lookback(self) -> int:
        return self.window + 1

    @property
    def horizon(self) -> int:
        return self._horizon

    def predict(self, listing_data: dict[str, np.ndarray]) -> float | None:
        bid0 = listing_data.get("bidPrice0")
        ask0 = listing_data.get("askPrice0")

        if bid0 is None or ask0 is None:
            return None
        if len(bid0) < self.min_lookback or len(ask0) < self.min_lookback:
            return None

        mid = (bid0[-self.min_lookback:] + ask0[-self.min_lookback:]) / 2.0
        log_returns = np.diff(np.log(mid))

        # Rolling std over window with ddof=1 to match pandas convention
        vol = float(np.std(log_returns[-self.window:], ddof=1))

        # Convert per-tick sigma to bps
        return vol * 1e4
