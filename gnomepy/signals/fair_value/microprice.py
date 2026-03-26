from __future__ import annotations

from pathlib import Path

from gnomepy.java.schemas import JavaSchema
from gnomepy.signals.fair_value.base import FairValueModel


class MicropriceFairValue(FairValueModel):
    """L1 microprice: volume-weighted mid using top-of-book.

    microprice = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)

    When ask_size >> bid_size (selling pressure), microprice shifts toward bid.
    When bid_size >> ask_size (buying pressure), microprice shifts toward ask.

    Args:
        alpha: EWMA smoothing factor (0.99 = slow, 0.9 = fast). 1.0 = no smoothing.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._fair_value = 0
        self._ready = False

    def update(self, timestamp: int, data: JavaSchema) -> None:
        bid = data.bid_price(0)
        ask = data.ask_price(0)
        bid_size = data.bid_size(0)
        ask_size = data.ask_size(0)

        if bid <= 0 or ask <= 0 or bid_size + ask_size <= 0:
            return

        raw = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)

        if not self._ready:
            self._fair_value = raw
            self._ready = True
        else:
            self._fair_value = self.alpha * self._fair_value + (1 - self.alpha) * raw

    def get_fair_value(self) -> int:
        return int(self._fair_value)

    def is_ready(self) -> bool:
        return self._ready

    def get_name(self) -> str:
        return "microprice-fair-value"

    def get_version(self) -> str:
        return "v1.0"

    def save_model(self, directory: Path):
        pass

    def load_model(self, directory: Path):
        pass

    def reset(self):
        self._fair_value = 0
        self._ready = False




class WeightedMicropriceFairValue(FairValueModel):
    """Multi-level weighted microprice using MBP book depth.

    Uses levels 0..num_levels from MBP10 data. Deeper levels are weighted
    less via exponential decay.

    weighted_microprice = Σ(w_i * (bid_i * ask_size_i + ask_i * bid_size_i)) /
                          Σ(w_i * (bid_size_i + ask_size_i))

    where w_i = decay ^ i

    Args:
        num_levels: Number of book levels to use (1-10).
        decay: Weight decay per level (0.5 = each level half the weight of previous).
        alpha: EWMA smoothing factor. 1.0 = no smoothing.
    """

    def __init__(self, num_levels: int = 5, decay: float = 0.5, alpha: float = 1.0):
        self.num_levels = min(num_levels, 10)
        self.decay = decay
        self.alpha = alpha
        self._fair_value = 0
        self._ready = False

        # Precompute weights
        self._weights = [decay ** i for i in range(self.num_levels)]

    def update(self, timestamp: int, data: JavaSchema) -> None:
        numerator = 0.0
        denominator = 0.0

        for i in range(self.num_levels):
            bid = data.bid_price(i)
            ask = data.ask_price(i)
            bid_size = data.bid_size(i)
            ask_size = data.ask_size(i)

            if bid <= 0 or ask <= 0 or bid_size + ask_size <= 0:
                continue

            w = self._weights[i]
            numerator += w * (bid * ask_size + ask * bid_size)
            denominator += w * (bid_size + ask_size)

        if denominator <= 0:
            return

        raw = numerator / denominator

        if not self._ready:
            self._fair_value = raw
            self._ready = True
        else:
            self._fair_value = self.alpha * self._fair_value + (1 - self.alpha) * raw

    def get_fair_value(self) -> int:
        return int(self._fair_value)

    def is_ready(self) -> bool:
        return self._ready

    def get_name(self) -> str:
        return "weighted-microprice-fair-value"

    def get_version(self) -> str:
        return "v1.0"

    def save_model(self, directory: Path):
        pass

    def load_model(self, directory: Path):
        pass

    def reset(self):
        self._fair_value = 0
        self._ready = False
