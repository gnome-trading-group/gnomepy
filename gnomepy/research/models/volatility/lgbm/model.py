"""
LGBMVolatilityModel — LGBM regression model for volatility prediction.

Uses dedicated volatility features to predict forward realized per-tick
volatility in bps.  Model outputs are in log1p space and transformed back
via expm1.  The per-tick normalization (sqrt(horizon)) is applied during
training so predictions are horizon-invariant.
"""

import json
from pathlib import Path

import numpy as np

from gnomepy.research.models.base import RegistrableModel
from gnomepy.research.models.volatility.lgbm.features import (
    VOL_FEATURE_NAMES,
    MIN_LOOKBACK,
    extract_vol_features_tick,
)
from gnomepy.research.models.registry import ModelRegistry
from gnomepy.research.models.volatility.base import VolatilityModel


class LGBMVolatilityModel(VolatilityModel, RegistrableModel):
    """LGBM regression model predicting forward realized per-tick volatility in bps.

    Uses 30 dedicated volatility features. Model outputs are in log1p space
    and transformed back via expm1 to produce per-tick bps predictions.

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
        import lightgbm as lgb

        self.model: lgb.Booster | None = None
        self._feature_names: list[str] = VOL_FEATURE_NAMES
        self._horizon = horizon
        self.listing_id: int | None = listing_id

        if model_path is not None:
            loaded = LGBMVolatilityModel.load_from_dir(Path(model_path))
            self.model = loaded.model
            self._feature_names = loaded._feature_names
            self._horizon = loaded._horizon
        elif registry is not None and listing_id is not None:
            self._load_from_registry(registry, listing_id, version)

    @property
    def model_type(self) -> str:
        return "volatility"

    @property
    def model_name(self) -> str:
        if self.listing_id is None:
            raise ValueError("listing_id must be set before accessing model_name")
        return f"listing_{self.listing_id}_h{self._horizon}"

    @property
    def min_lookback(self) -> int:
        return MIN_LOOKBACK

    @property
    def horizon(self) -> int:
        return self._horizon

    def predict(self, listing_data: dict[str, np.ndarray]) -> float | None:
        if self.model is None:
            return None

        features = extract_vol_features_tick(listing_data)
        if features is None:
            return None

        raw_pred = float(self.model.predict(features.reshape(1, -1))[0])
        return max(0.0, float(np.expm1(raw_pred)))

    # ------------------------------------------------------------------
    # RegistrableModel
    # ------------------------------------------------------------------

    def save_to_dir(self, path: Path) -> None:
        self.model.save_model(str(path / "model.txt"))

    @classmethod
    def load_from_dir(
        cls, path: Path, metadata: dict | None = None
    ) -> "LGBMVolatilityModel":
        import lightgbm as lgb

        if metadata is None:
            meta_path = path / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    metadata = json.load(f)
        metadata = metadata or {}

        instance = cls.__new__(cls)
        instance.model = lgb.Booster(model_file=str(path / "model.txt"))
        instance._feature_names = metadata.get("feature_names", VOL_FEATURE_NAMES)
        instance._horizon = int(metadata.get("horizon", 20))
        instance.listing_id = None
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_from_registry(
        self, registry: ModelRegistry, listing_id: int, version: str | None
    ) -> None:
        name = f"listing_{listing_id}_h{self._horizon}"
        if version is not None:
            loaded, _ = registry.load(self.model_type, name, version)
        else:
            loaded, _ = registry.load_latest(self.model_type, name)
        assert isinstance(loaded, LGBMVolatilityModel)
        self.model = loaded.model
        self._feature_names = loaded._feature_names
        self._horizon = loaded._horizon
