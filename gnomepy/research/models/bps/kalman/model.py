"""Kalman filter BPS prediction model."""

import json
from pathlib import Path

import numpy as np

from gnomepy.research.models.base import RegistrableModel
from gnomepy.research.models.bps.base import BpsModel


class KalmanBpsModel(BpsModel, RegistrableModel):
    """Kalman filter model for expected forward return prediction.

    Uses a local-linear-trend filter on log mid-prices sampled every
    ``sample_interval`` ticks.  The state vector is ``[level, trend]``
    where *level* tracks the current log mid-price and *trend* captures
    the per-sample drift.  Observing log-prices rather than log-returns
    avoids zero-inflation from price discreteness while keeping the
    series scale-invariant across instruments.

    Filter state is carried across ``predict()`` calls so the filter
    builds conviction over time rather than re-initialising from a
    diffuse prior on every invocation.

    Parameters
    ----------
    window : int
        Minimum number of sampled observations needed before the first
        prediction (filter warm-up).
    horizon : int
        Forward horizon in ticks.  The per-sample trend is converted
        to a per-tick rate and multiplied by this horizon.
    sample_interval : int
        Observe log(mid) every *sample_interval* ticks instead of
        every tick.
    observation_noise : float
        Observation noise variance (R).  Captures microstructure
        noise in the mid-price (bid-ask bounce, etc.).
    process_noise_level : float
        Process noise scaling factor (q).
    """

    def __init__(
        self,
        window: int = 200,
        horizon: int = 20,
        sample_interval: int = 10,
        observation_noise: float = 1e-6,
        process_noise_level: float = 1e-8,
    ):
        self.window = window
        self._horizon = horizon
        self.sample_interval = sample_interval
        self.observation_noise = observation_noise
        self.process_noise_level = process_noise_level
        self.reset()

    # ------------------------------------------------------------------
    # BpsModel
    # ------------------------------------------------------------------

    @property
    def min_lookback(self) -> int:
        return self.window * self.sample_interval + 1

    def predict(self, listing_data: dict[str, np.ndarray]) -> float | None:
        bid0 = listing_data.get("bidPrice0")
        ask0 = listing_data.get("askPrice0")

        if bid0 is None or ask0 is None:
            return None

        n = len(bid0)
        if n < self.min_lookback:
            return None

        # If the data is shorter than what we've seen, the listing
        # changed — reset and start fresh.
        if n < self._ticks_seen:
            self.reset()

        mid = (bid0 + ask0) / 2.0

        # Pre-compute filter matrices
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        q = self.process_noise_level
        Q = q * np.array([[1.0 / 3.0, 0.5], [0.5, 1.0]])
        R = self.observation_noise

        start = self._ticks_seen
        for i in range(start, n):
            self._ticks_seen += 1
            if self._ticks_seen % self.sample_interval == 0:
                z = np.log(float(mid[i]))
                if not self._initialized:
                    # Seed level from first observation; trend unknown
                    self._x[0] = z
                    self._initialized = True
                    self._n_updates += 1
                    continue
                # Predict
                self._x = F @ self._x
                self._P = F @ self._P @ F.T + Q
                # Update
                innov = z - self._x[0]
                S = self._P[0, 0] + R
                K = self._P[:, 0] / S
                self._x = self._x + K * innov
                self._P = self._P - np.outer(K, self._P[0, :])
                self._n_updates += 1

        if self._n_updates < 2:
            return None

        # trend is per-sample-interval; convert to per-tick, then scale
        # by forward horizon
        trend_per_tick = self._x[1] / self.sample_interval
        return float(trend_per_tick * self._horizon * 1e4)

    # ------------------------------------------------------------------
    # RegistrableModel
    # ------------------------------------------------------------------

    @property
    def model_type(self) -> str:
        return "bps"

    @property
    def model_name(self) -> str:
        return (
            f"kalman_w{self.window}_h{self._horizon}"
            f"_s{self.sample_interval}"
            f"_r{self.observation_noise:.0e}"
            f"_q{self.process_noise_level:.0e}"
        )

    def save_to_dir(self, path: Path) -> None:
        pass

    @classmethod
    def load_from_dir(
        cls, path: Path, metadata: dict | None = None
    ) -> "KalmanBpsModel":
        if metadata is None:
            with open(path / "metadata.json") as f:
                metadata = json.load(f)
        return cls(
            window=metadata.get("window", 200),
            horizon=metadata.get("horizon", 20),
            sample_interval=metadata.get("sample_interval", 10),
            observation_noise=metadata.get("observation_noise", 1e-6),
            process_noise_level=metadata.get("process_noise_level", 1e-8),
        )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear filter state.  Call when switching listings."""
        self._x = np.zeros(2)
        self._P = 1e4 * np.eye(2)
        self._ticks_seen: int = 0
        self._n_updates: int = 0
        self._initialized: bool = False
