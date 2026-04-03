from __future__ import annotations

from pathlib import Path

from gnomepy.java.schemas import Schema
from gnomepy.signals.fair_value.base import FairValueModel


class MidFairValue(FairValueModel):
    """Fair value = simple mid price. Default behavior."""

    def __init__(self):
        self._mid = 0

    def update(self, timestamp: int, data: Schema) -> None:
        bid = data.bid_price(0)
        ask = data.ask_price(0)
        if bid > 0 and ask > 0:
            self._mid = (bid + ask) // 2

    def get_fair_value(self) -> int:
        return self._mid

    def is_ready(self) -> bool:
        return self._mid > 0

    def get_name(self) -> str:
        return "mid-fair-value"

    def get_version(self) -> str:
        return "v1.0"

    def save_model(self, directory: Path):
        pass

    def load_model(self, directory: Path):
        pass

    def reset(self):
        self._mid = 0
