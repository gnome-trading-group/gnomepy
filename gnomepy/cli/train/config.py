"""Pydantic models for training configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class TrainingDataConfig(BaseModel):
    listing_id: int
    start_datetime: str
    end_datetime: str
    schema_type: str = "mbp-10"


class LGBMVolTrainerConfig(BaseModel):
    type: Literal["lgbm_volatility"]
    horizon: int = Field(default=20, gt=0)


TrainerConfig = Annotated[LGBMVolTrainerConfig, Field(discriminator="type")]


class TuningConfig(BaseModel):
    enabled: bool = True
    train_window: int = 50000
    val_window: int = 10000
    n_random: int | None = None
    metric: str = "mae"
    num_boost_round: int = 500
    param_grid: dict[str, list] | None = None


class TrainExecutionConfig(BaseModel):
    num_boost_round: int = 500
    early_stopping_rounds: int = 50
    params: dict | None = None


class TrainRegistryConfig(BaseModel):
    base_dir: str = "./models"
    upload_s3: bool = False
    s3_bucket: str | None = None
    s3_prefix: str = "models/"


class TrainConfig(BaseModel):
    data: TrainingDataConfig
    model: TrainerConfig
    tuning: TuningConfig = TuningConfig()
    training: TrainExecutionConfig = TrainExecutionConfig()
    registry: TrainRegistryConfig = TrainRegistryConfig()


def load_train_config(path: str) -> TrainConfig:
    from gnomepy.cli.config import load_yaml

    data = load_yaml(path)
    return TrainConfig.model_validate(data)
