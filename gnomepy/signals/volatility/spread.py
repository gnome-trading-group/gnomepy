from __future__ import annotations

from pathlib import Path

from gnomepy.java.schemas import Schema
from gnomepy.signals.volatility.base import VolatilitySignal


class SpreadVolatility(VolatilitySignal):
    """Volatility estimate derived from the bid-ask spread.

    The spread reflects the market's implied short-term volatility — wider spreads
    mean market makers expect more price uncertainty. This signal uses an exponentially
    weighted moving average of the spread in bps as the volatility estimate.

    Args:
        alpha: EWMA decay factor (0.99 = slow, 0.9 = fast).
        warmup_ticks: Minimum ticks before the signal is ready.
        scale: Multiplier on the spread to convert to vol estimate.
    """

    def __init__(
        self,
        alpha: float = 0.95,
        warmup_ticks: int = 50,
        scale: float = 1.0,
    ):
        self.alpha = alpha
        self.warmup_ticks = warmup_ticks
        self.scale = scale
        self._ewma_spread_bps = 0.0
        self._tick_count = 0

    def update(self, timestamp: int, data: Schema):
        bid = data.bid_price(0)
        ask = data.ask_price(0)
        if bid <= 0 or ask <= 0:
            return

        mid = (bid + ask) / 2
        spread_bps = (ask - bid) / mid * 10000

        if self._tick_count == 0:
            self._ewma_spread_bps = spread_bps
        else:
            self._ewma_spread_bps = self.alpha * self._ewma_spread_bps + (1 - self.alpha) * spread_bps

        self._tick_count += 1

    def get_prediction(self) -> float:
        return self._ewma_spread_bps * self.scale

    def is_ready(self) -> bool:
        return self._tick_count >= self.warmup_ticks

    def reset(self):
        self._ewma_spread_bps = 0.0
        self._tick_count = 0

    def get_name(self) -> str:
        return "spread-volatility"

    def get_version(self) -> str:
        return "v1.0"

    def save_model(self, directory: Path):
        pass

    def load_model(self, directory: Path):
        pass
