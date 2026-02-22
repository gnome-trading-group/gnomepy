"""
LGBMVolatilityModel — LGBM regression model for volatility prediction.

Reuses feature extraction from lgbm_directional to predict forward
absolute price movement in bps. Used as a VolatilityModel for the
BasicMMSignal circuit breaker.
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from gnomepy.research.signals.lgbm_directional.features import (
    FEATURE_NAMES,
    MIN_LOOKBACK,
    extract_features_tick,
)
from gnomepy.research.signals.lgbm_directional.registry import ModelRegistry

from ..volatility_model import VolatilityModel


class LGBMVolatilityModel(VolatilityModel):
    """LGBM regression model predicting forward absolute movement in bps.

    Uses the same 44 features as the directional model but trained with
    regression objective on absolute returns.

    Model loading supports two modes:

    * **Direct path**: ``model_path="./vol_models/lgbm_btc/v1"`` -- loads
      ``model.txt`` + ``metadata.json`` from the directory.
    * **Via registry**: ``registry=ModelRegistry(base_dir="./vol_models"), listing_id=1``
      with optional ``version``.

    Parameters
    ----------
    model_path : str, optional
        Path to a version directory containing ``model.txt`` and
        ``metadata.json``.
    registry : ModelRegistry, optional
        Registry instance for model loading.
    listing_id : int, optional
        Listing ID for registry loading.
    version : str, optional
        Model version to load from registry. ``None`` loads latest.
    """

    def __init__(
        self,
        model_path: str | None = None,
        registry: ModelRegistry | None = None,
        listing_id: int | None = None,
        version: str | None = None,
        horizon: int = 20,
    ):
        self.model: lgb.Booster | None = None
        self._feature_names: list[str] = FEATURE_NAMES
        self._horizon = horizon

        if model_path is not None:
            self._load_from_path(model_path)
        elif registry is not None and listing_id is not None:
            self._load_from_registry(registry, listing_id, version)

    @property
    def min_lookback(self) -> int:
        return MIN_LOOKBACK

    @property
    def horizon(self) -> int:
        return self._horizon

    def predict(self, listing_data: dict[str, np.ndarray]) -> float | None:
        if self.model is None:
            return None

        features = extract_features_tick(listing_data)
        if features is None:
            return None

        pred = float(self.model.predict(features.reshape(1, -1))[0])
        return max(0.0, pred)

    def _load_from_path(self, model_path: str) -> None:
        path = Path(model_path)
        self.model = lgb.Booster(model_file=str(path / "model.txt"))
        meta_path = path / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self._feature_names = meta.get("feature_names", FEATURE_NAMES)
            if "horizon" in meta:
                self._horizon = int(meta["horizon"])

    def _load_from_registry(
        self, registry: ModelRegistry, listing_id: int, version: str | None
    ) -> None:
        if version is not None:
            model, meta = registry.load(listing_id, version)
        else:
            model, meta = registry.load_latest(listing_id)
        self.model = model
        self._feature_names = meta.get("feature_names", FEATURE_NAMES)
        if "horizon" in meta:
            self._horizon = int(meta["horizon"])
