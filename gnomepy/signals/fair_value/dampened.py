from __future__ import annotations

from pathlib import Path

from gnomepy.java.schemas import Schema
from gnomepy.signals.fair_value.base import FairValueModel


class DampenedFairValue(FairValueModel):
    """Wraps any FairValueModel and blends its output toward mid.

    fair_value = mid + damping * (inner_model.get_fair_value() - mid)

    Reduces directional bias from aggressive fair value models without
    losing the signal entirely.

    At damping=1.0: pure inner model (passthrough).
    At damping=0.1: 90% mid, 10% inner model signal.
    At damping=0.0: pure mid (ignores inner model).

    Args:
        model: Any FairValueModel to dampen.
        damping: Blend factor (0.0-1.0). Lower = more conservative.
    """

    def __init__(self, model: FairValueModel, damping: float = 0.1):
        self.model = model
        self.damping = damping
        self._mid = 0
        self._fair_value = 0
        self._ready = False

    def update(self, timestamp: int, data: Schema) -> None:
        bid = data.bid_price(0)
        ask = data.ask_price(0)
        if bid <= 0 or ask <= 0:
            return

        self._mid = (bid + ask) // 2
        self.model.update(timestamp, data)

        if self.model.is_ready():
            inner = self.model.get_fair_value()
            self._fair_value = int(self._mid + self.damping * (inner - self._mid))
            self._ready = True

    def get_fair_value(self) -> int:
        return self._fair_value

    def is_ready(self) -> bool:
        return self._ready

    def get_name(self) -> str:
        return f"dampened-{self.model.get_name()}"

    def get_version(self) -> str:
        return self.model.get_version()

    def save_model(self, directory: Path):
        self.model.save_model(directory)

    def load_model(self, directory: Path):
        self.model.load_model(directory)

    def reset(self):
        self.model.reset()
        self._mid = 0
        self._fair_value = 0
        self._ready = False
