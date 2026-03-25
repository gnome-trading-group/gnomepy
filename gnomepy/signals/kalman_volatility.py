from __future__ import annotations

import json
import math
from pathlib import Path

from gnomepy.java.schemas import JavaSchema
from gnomepy.signals.volatility_signal import VolatilitySignal


class AdaptiveKalmanVolatility(VolatilitySignal):
    """Adaptive Kalman filter for volatility estimation.

    Models log-return variance as a hidden state with a random walk transition.
    Measurement noise R is adapted online from innovations, making the filter
    responsive to regime changes.

    Output: volatility in basis points.

    Args:
        Q: Process noise variance — how fast true volatility changes between ticks.
        alpha: R adaptation smoothing (0-1). Higher = slower adaptation.
        r_floor: Minimum measurement noise — prevents R collapse.
        warmup_ticks: Ticks before is_ready() returns True.
        version: Model version string.
    """

    def __init__(
        self,
        Q: float = 1e-8,
        alpha: float = 0.99,
        r_floor: float = 1e-10,
        warmup_ticks: int = 100,
        version: str = "v1.0",
    ):
        self.Q = Q
        self.alpha = alpha
        self.r_floor = r_floor
        self.warmup_ticks = warmup_ticks
        self._version = version
        self.reset()

    def update(self, timestamp: int, data: JavaSchema):
        bid = data.bid_price(0)
        ask = data.ask_price(0)
        if bid <= 0 or ask <= 0:
            return

        mid = (bid + ask) // 2

        if self._last_price <= 0:
            self._last_price = mid
            return

        # Log return approximation
        log_return = (mid - self._last_price) / self._last_price
        self._last_price = mid

        # Observation: squared return
        z = log_return * log_return
        self._tick_count += 1

        if self._tick_count == 1:
            self._x = z
            self._P = z
            self._R = z
            return

        # Predict
        x_pred = self._x
        p_pred = self._P + self.Q

        # Innovation
        innovation = z - x_pred
        S = p_pred + self._R

        # Adaptive R
        r_candidate = innovation * innovation - p_pred
        self._R = self.alpha * self._R + (1.0 - self.alpha) * max(r_candidate, self.r_floor)

        # Gain
        K = p_pred / S if S > 0 else 0.0

        # Update
        self._x = x_pred + K * innovation
        self._P = (1.0 - K) * p_pred

        if self._x < 0:
            self._x = 0
        if self._P < 0:
            self._P = self.r_floor

    def get_prediction(self) -> float:
        return math.sqrt(max(self._x, 0)) * 10000.0

    def is_ready(self) -> bool:
        return self._tick_count >= self.warmup_ticks

    def reset(self):
        self._x = 0.0
        self._P = 0.0
        self._R = self.r_floor
        self._last_price = 0
        self._tick_count = 0

    def save_model(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        data = {
            "Q": self.Q,
            "alpha": self.alpha,
            "r_floor": self.r_floor,
            "warmup_ticks": self.warmup_ticks,
            "x": self._x,
            "P": self._P,
            "R": self._R,
            "last_price": self._last_price,
            "tick_count": self._tick_count,
        }
        (directory / "params.json").write_text(json.dumps(data))

    def load_model(self, directory: Path):
        data = json.loads((directory / "params.json").read_text())
        self._x = data["x"]
        self._P = data["P"]
        self._R = data["R"]
        self._last_price = data["last_price"]
        self._tick_count = data["tick_count"]

    def get_name(self) -> str:
        return "adaptive-kalman-volatility"

    def get_version(self) -> str:
        return self._version

    # Inspection
    @property
    def variance_estimate(self) -> float:
        return self._x

    @property
    def covariance(self) -> float:
        return self._P

    @property
    def measurement_noise(self) -> float:
        return self._R

    @property
    def kalman_gain(self) -> float:
        denom = self._P + self.Q + self._R
        return (self._P + self.Q) / denom if denom > 0 else 0.0

    @property
    def tick_count(self) -> int:
        return self._tick_count
