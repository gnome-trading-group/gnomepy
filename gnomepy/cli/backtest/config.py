"""Pydantic models for backtest configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

class StaticLatencyConfig(BaseModel):
    type: Literal["static"]
    latency_ns: int = Field(ge=0)


class GaussianLatencyConfig(BaseModel):
    type: Literal["gaussian"]
    mu: float = Field(ge=0)
    sigma: float = Field(gt=0)


LatencyConfig = Annotated[
    StaticLatencyConfig | GaussianLatencyConfig,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Fee
# ---------------------------------------------------------------------------

class StaticFeeConfig(BaseModel):
    type: Literal["static"]
    taker_fee: float = Field(ge=0)
    maker_fee: float = Field(ge=0)


FeeModelConfig = Annotated[
    StaticFeeConfig,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------

class RiskAverseQueueConfig(BaseModel):
    type: Literal["risk_averse"]


class OptimisticQueueConfig(BaseModel):
    type: Literal["optimistic"]


class ProbabilisticQueueConfig(BaseModel):
    type: Literal["probabilistic"]
    cancel_ahead_probability: float = Field(default=0.5, ge=0, le=1)


QueueModelConfig = Annotated[
    RiskAverseQueueConfig | OptimisticQueueConfig | ProbabilisticQueueConfig,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Volatility model
# ---------------------------------------------------------------------------

class RealizedVolConfig(BaseModel):
    type: Literal["realized"]
    window: int = Field(default=100, gt=0)
    horizon: int = Field(default=20, gt=0)


class LGBMVolConfig(BaseModel):
    type: Literal["lgbm"]
    model_path: str | None = None
    horizon: int = Field(default=20, gt=0)


VolatilityModelConfig = Annotated[
    RealizedVolConfig | LGBMVolConfig,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# BPS model
# ---------------------------------------------------------------------------

class KalmanBpsConfig(BaseModel):
    type: Literal["kalman"]
    window: int = Field(default=200, gt=0)
    horizon: int = Field(default=20, gt=0)
    sample_interval: int = Field(default=10, gt=0)
    observation_noise: float = Field(default=1e-6, gt=0)
    process_noise_level: float = Field(default=1e-8, gt=0)


BpsModelConfig = Annotated[
    KalmanBpsConfig,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

class MarketMakingSignalConfig(BaseModel):
    type: Literal["market_making"]
    listing_id: int
    volatility_model: VolatilityModelConfig
    vol_threshold_bps: float = 10.0
    trade_frequency: int = Field(default=1, gt=0)
    gamma: float = 0.1
    order_arrival_rate: float = 1.0
    volatility_window: int = 100
    volatility_half_life: float = 0.5
    max_inventory: float | None = None
    liquidation_threshold: float = 0.8
    use_market_orders_for_liquidation: bool = True
    min_spread_bps: float | None = None
    max_spread_ticks: float | None = None
    min_volatility: float = 1e-8


class BpsSignalConfig(BaseModel):
    type: Literal["bps"]
    listing_id: int
    model: BpsModelConfig | None = None
    trade_frequency: int = Field(default=1, gt=0)
    entry_threshold_bps: float = 5.0
    transaction_cost_bps: float = 2.0
    max_inventory: float = 100.0
    exit_hold_ticks: int = 100
    stop_loss_bps: float = 10.0
    take_profit_bps: float = 15.0


SignalConfig = Annotated[
    MarketMakingSignalConfig | BpsSignalConfig,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# OMS
# ---------------------------------------------------------------------------

class SimpleOMSConfig(BaseModel):
    type: Literal["simple"]
    notional: float = Field(default=1000.0, gt=0)
    starting_cash: float = 1000000.0


class MarketMakingOMSConfig(BaseModel):
    type: Literal["market_making"]
    notional: float = Field(default=1000.0, gt=0)
    starting_cash: float = 1000000.0
    max_position_notional: float | None = None
    position_aware_sizing: bool = True
    position_scaling_factor: float = 0.5
    tick_size: float = Field(default=0.01, gt=0)
    passive_reprice_ticks: int = Field(default=3, ge=0)


OMSConfig = Annotated[
    SimpleOMSConfig | MarketMakingOMSConfig,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Exchange
# ---------------------------------------------------------------------------

class ExchangeDefaultConfig(BaseModel):
    fee_model: FeeModelConfig
    network_latency: LatencyConfig
    order_processing_latency: LatencyConfig
    queue_model: QueueModelConfig


class ExchangeConfig(BaseModel):
    default: ExchangeDefaultConfig


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class StrategyConfig(BaseModel):
    type: Literal["simple"] = "simple"
    processing_latency: LatencyConfig = StaticLatencyConfig(type="static", latency_ns=0)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

class OutputConfig(BaseModel):
    s3_bucket: str | None = None
    s3_prefix: str = "backtests/"


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

class BacktestTimeConfig(BaseModel):
    start_datetime: str
    end_datetime: str
    listing_ids: list[int] = Field(min_length=1)
    schema_type: str


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

class BacktestConfig(BaseModel):
    backtest: BacktestTimeConfig
    exchange: ExchangeConfig
    strategy: StrategyConfig = StrategyConfig()
    oms: OMSConfig
    signals: list[SignalConfig] = Field(min_length=1)
    output: OutputConfig = OutputConfig()

    @model_validator(mode="after")
    def _check_signal_listing_ids(self) -> "BacktestConfig":
        valid_ids = set(self.backtest.listing_ids)
        for i, sig in enumerate(self.signals):
            if sig.listing_id not in valid_ids:
                raise ValueError(
                    f"signals[{i}].listing_id={sig.listing_id} "
                    f"not in backtest.listing_ids={self.backtest.listing_ids}"
                )
        return self


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> BacktestConfig:
    from gnomepy.cli.config import load_yaml

    data = load_yaml(path)
    return BacktestConfig.model_validate(data)
