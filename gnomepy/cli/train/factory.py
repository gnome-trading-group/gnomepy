"""Build a trainer from a validated training config."""

import datetime

from gnomepy.cli.train.config import TrainConfig
from gnomepy.data.types import SchemaType
from gnomepy.research.models.volatility.lgbm.trainer import LGBMVolatilityTrainer


def build_trainer(config: TrainConfig) -> LGBMVolatilityTrainer:
    """Construct the appropriate trainer from the config."""
    match config.model.type:
        case "lgbm_volatility":
            return LGBMVolatilityTrainer(
                listing_id=config.data.listing_id,
                start_datetime=datetime.datetime.fromisoformat(config.data.start_datetime),
                end_datetime=datetime.datetime.fromisoformat(config.data.end_datetime),
                schema_type=SchemaType(config.data.schema_type),
                horizon=config.model.horizon,
            )
        case _:
            raise ValueError(f"Unknown model type: {config.model.type}")
