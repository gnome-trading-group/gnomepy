"""Simple realized volatility model."""

import json
from pathlib import Path

import numpy as np

from gnomepy.research.models.base import RegistrableModel
from gnomepy.research.models.volatility.base import VolatilityModel


class RealizedVolatilityModel(VolatilityModel, RegistrableModel):
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
    def model_type(self) -> str:
        return "volatility"

    @property
    def model_name(self) -> str:
        return f"realized_w{self.window}_h{self._horizon}"

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

    # ------------------------------------------------------------------
    # RegistrableModel
    # ------------------------------------------------------------------

    def save_to_dir(self, path: Path) -> None:
        # Params-only model — no artifacts to write; metadata.json is
        # written by the registry itself.
        pass

    @classmethod
    def load_from_dir(
        cls, path: Path, metadata: dict | None = None
    ) -> "RealizedVolatilityModel":
        if metadata is None:
            with open(path / "metadata.json") as f:
                metadata = json.load(f)
        return cls(
            window=metadata.get("window", 100),
            horizon=metadata.get("horizon", 20),
        )
