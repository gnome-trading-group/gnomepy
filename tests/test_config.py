"""Unit tests for BacktestConfig dataclass construction — no JVM needed."""
from __future__ import annotations

from datetime import date, datetime

import pytest

from gnomepy.java.backtest.config import (
    BacktestConfig,
    ExchangeProfileConfig,
    GaussianLatencyConfig,
    ListingSimConfig,
    OptimisticQueueConfig,
    ProbabilisticQueueConfig,
    RiskAverseQueueConfig,
    RiskConfig,
    S3Config,
    StaticFeeConfig,
    StaticLatencyConfig,
    StrategyConfig,
)


class TestStaticFeeConfig:
    def test_defaults(self):
        cfg = StaticFeeConfig()
        assert cfg.taker_fee == 0.0
        assert cfg.maker_fee == 0.0

    def test_explicit_values(self):
        cfg = StaticFeeConfig(taker_fee=0.0005, maker_fee=-0.0001)
        assert cfg.taker_fee == 0.0005
        assert cfg.maker_fee == -0.0001


class TestLatencyConfigs:
    def test_static_latency_default(self):
        cfg = StaticLatencyConfig()
        assert cfg.latency_nanos == 0

    def test_static_latency_explicit(self):
        cfg = StaticLatencyConfig(latency_nanos=5_000_000)
        assert cfg.latency_nanos == 5_000_000

    def test_gaussian_latency(self):
        cfg = GaussianLatencyConfig(mu=1_000_000.0, sigma=200_000.0)
        assert cfg.mu == 1_000_000.0
        assert cfg.sigma == 200_000.0


class TestQueueConfigs:
    def test_optimistic_instantiates(self):
        assert OptimisticQueueConfig() is not None

    def test_risk_averse_instantiates(self):
        assert RiskAverseQueueConfig() is not None

    def test_probabilistic_default(self):
        cfg = ProbabilisticQueueConfig()
        assert cfg.cancel_ahead_probability == 0.5

    def test_probabilistic_explicit(self):
        cfg = ProbabilisticQueueConfig(cancel_ahead_probability=0.3)
        assert cfg.cancel_ahead_probability == 0.3


class TestExchangeProfileConfig:
    def test_defaults(self):
        cfg = ExchangeProfileConfig()
        assert isinstance(cfg.fee_model, StaticFeeConfig)
        assert isinstance(cfg.network_latency, StaticLatencyConfig)
        assert isinstance(cfg.order_processing_latency, StaticLatencyConfig)
        assert isinstance(cfg.queue_model, RiskAverseQueueConfig)

    def test_custom_profile(self):
        cfg = ExchangeProfileConfig(
            fee_model=StaticFeeConfig(taker_fee=0.001, maker_fee=-0.0002),
            network_latency=GaussianLatencyConfig(mu=5e6, sigma=1e6),
            queue_model=OptimisticQueueConfig(),
        )
        assert cfg.fee_model.taker_fee == 0.001
        assert isinstance(cfg.network_latency, GaussianLatencyConfig)
        assert isinstance(cfg.queue_model, OptimisticQueueConfig)


class TestListingSimConfig:
    def test_fields(self):
        cfg = ListingSimConfig(listing_id=42, profile="default")
        assert cfg.listing_id == 42
        assert cfg.profile == "default"


class TestStrategyConfig:
    def test_python_path(self):
        cfg = StrategyConfig(class_name="my.module:MyStrategy")
        assert cfg.class_name == "my.module:MyStrategy"
        assert cfg.args is None

    def test_with_args(self):
        cfg = StrategyConfig(class_name="my.module:MyStrategy", args={"size": 100})
        assert cfg.args == {"size": 100}


class TestRiskConfig:
    def test_empty_default(self):
        cfg = RiskConfig()
        assert cfg.policies == {}

    def test_with_policies(self):
        cfg = RiskConfig(policies={"MAX_NOTIONAL": {"maxNotionalValue": 1_000_000}})
        assert "MAX_NOTIONAL" in cfg.policies


class TestS3Config:
    def test_default_bucket(self):
        cfg = S3Config()
        assert cfg.bucket == "gnome-market-data-prod"

    def test_custom_bucket(self):
        cfg = S3Config(bucket="my-bucket")
        assert cfg.bucket == "my-bucket"


class TestBacktestConfig:
    def _minimal(self):
        return BacktestConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 2),
            listings=[ListingSimConfig(listing_id=1, profile="default")],
            profiles={"default": ExchangeProfileConfig()},
        )

    def test_minimal_construction(self):
        cfg = self._minimal()
        assert cfg.start_date == date(2024, 1, 1)
        assert len(cfg.listings) == 1
        assert "default" in cfg.profiles

    def test_defaults(self):
        cfg = self._minimal()
        assert cfg.strategy is None
        assert isinstance(cfg.risk, RiskConfig)
        assert isinstance(cfg.s3, S3Config)
        assert cfg.record is True
        assert cfg.record_depth == 1

    def test_datetime_start_end(self):
        cfg = BacktestConfig(
            start_date=datetime(2024, 6, 1, 10, 30),
            end_date=datetime(2024, 6, 1, 11, 0),
            listings=[ListingSimConfig(listing_id=1, profile="p")],
            profiles={"p": ExchangeProfileConfig()},
        )
        assert isinstance(cfg.start_date, datetime)
        assert cfg.start_date.hour == 10

    def test_with_strategy(self):
        cfg = BacktestConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 2),
            listings=[ListingSimConfig(listing_id=1, profile="default")],
            profiles={"default": ExchangeProfileConfig()},
            strategy=StrategyConfig(class_name="gnomepy_research.strategies.momentum:MomentumTaker"),
        )
        assert cfg.strategy is not None
        assert ":" in cfg.strategy.class_name

    def test_record_false(self):
        cfg = self._minimal()
        cfg = BacktestConfig(
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            listings=cfg.listings,
            profiles=cfg.profiles,
            record=False,
        )
        assert cfg.record is False
