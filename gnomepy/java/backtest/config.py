from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Union

import jpype


@dataclass
class StaticFeeConfig:
    taker_fee: float = 0.0
    maker_fee: float = 0.0

    def _to_java(self):
        cls = jpype.JClass("group.gnometrading.backtest.config.FeeModelConfig$Static")
        obj = cls()
        obj.takerFee = float(self.taker_fee)
        obj.makerFee = float(self.maker_fee)
        return obj


@dataclass
class StaticLatencyConfig:
    latency_nanos: int = 0

    def _to_java(self):
        cls = jpype.JClass("group.gnometrading.backtest.config.LatencyConfig$Static")
        obj = cls()
        obj.latencyNanos = jpype.JLong(self.latency_nanos)
        return obj


@dataclass
class GaussianLatencyConfig:
    mu: float = 0.0
    sigma: float = 0.0

    def _to_java(self):
        cls = jpype.JClass("group.gnometrading.backtest.config.LatencyConfig$Gaussian")
        obj = cls()
        obj.mu = float(self.mu)
        obj.sigma = float(self.sigma)
        return obj


@dataclass
class OptimisticQueueConfig:
    def _to_java(self):
        return jpype.JClass("group.gnometrading.backtest.config.QueueModelConfig$Optimistic")()


@dataclass
class RiskAverseQueueConfig:
    def _to_java(self):
        return jpype.JClass("group.gnometrading.backtest.config.QueueModelConfig$RiskAverse")()


@dataclass
class ProbabilisticQueueConfig:
    cancel_ahead_probability: float = 0.5

    def _to_java(self):
        cls = jpype.JClass("group.gnometrading.backtest.config.QueueModelConfig$Probabilistic")
        obj = cls()
        obj.cancelAheadProbability = float(self.cancel_ahead_probability)
        return obj


@dataclass
class ExchangeProfileConfig:
    """Reusable simulation profile for a listing."""

    fee_model: StaticFeeConfig = field(default_factory=StaticFeeConfig)
    network_latency: Union[StaticLatencyConfig, GaussianLatencyConfig] = field(
        default_factory=StaticLatencyConfig
    )
    order_processing_latency: Union[StaticLatencyConfig, GaussianLatencyConfig] = field(
        default_factory=StaticLatencyConfig
    )
    queue_model: Union[
        OptimisticQueueConfig, RiskAverseQueueConfig, ProbabilisticQueueConfig
    ] = field(default_factory=RiskAverseQueueConfig)

    def _to_java(self):
        cls = jpype.JClass("group.gnometrading.backtest.config.ExchangeProfileConfig")
        obj = cls()
        obj.feeModel = self.fee_model._to_java()
        obj.networkLatency = self.network_latency._to_java()
        obj.orderProcessingLatency = self.order_processing_latency._to_java()
        obj.queueModel = self.queue_model._to_java()
        return obj


@dataclass
class ListingSimConfig:
    """Per-listing simulation config: a listing ID and the profile name to use."""

    listing_id: int
    profile: str

    def _to_java(self):
        cls = jpype.JClass("group.gnometrading.backtest.config.ListingSimConfig")
        obj = cls()
        obj.listingId = jpype.JInt(self.listing_id)
        obj.profile = str(self.profile)
        return obj


@dataclass
class StrategyConfig:
    """Strategy class name and constructor args for YAML-based strategy resolution."""

    class_name: str
    args: dict | None = None

    def _to_java(self):
        cls = jpype.JClass("group.gnometrading.backtest.config.StrategyConfig")
        HashMap = jpype.JClass("java.util.HashMap")
        obj = cls()
        obj.className = str(self.class_name)
        if self.args:
            m = HashMap()
            for k, v in self.args.items():
                m.put(str(k), v)
            obj.args = m
        return obj


@dataclass
class RiskConfig:
    """OMS risk policy configuration.

    Keys are RiskPolicyType enum names; values are parameter dicts.

    Example::

        RiskConfig(policies={
            "MAX_NOTIONAL": {"maxNotionalValue": 100_000},
            "MAX_ORDER_SIZE": {"maxOrderSize": 5_000},
            "KILL_SWITCH": {},
        })
    """

    policies: dict[str, dict[str, object]] = field(default_factory=dict)

    def _to_java(self):
        cls = jpype.JClass("group.gnometrading.backtest.config.RiskConfig")
        HashMap = jpype.JClass("java.util.HashMap")
        obj = cls()
        policies_map = HashMap()
        for policy_name, params in (self.policies or {}).items():
            inner = HashMap()
            for k, v in (params or {}).items():
                inner.put(str(k), v)
            policies_map.put(str(policy_name), inner)
        obj.policies = policies_map
        return obj


@dataclass
class S3Config:
    bucket: str = "gnome-market-data-prod"

    def _to_java(self):
        cls = jpype.JClass("group.gnometrading.backtest.config.S3Config")
        obj = cls()
        obj.bucket = str(self.bucket)
        return obj


@dataclass
class BacktestConfig:
    """Full backtest configuration.

    Usage::

        config = BacktestConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 2),
            listings=[ListingSimConfig(listing_id=1, profile="default")],
            profiles={"default": ExchangeProfileConfig()},
        )
        results = run_backtest(config, strategy=MyStrategy())
    """

    start_date: date | datetime
    end_date: date | datetime
    listings: list[ListingSimConfig]
    profiles: dict[str, ExchangeProfileConfig]
    strategy: StrategyConfig | None = None
    risk: RiskConfig = field(default_factory=RiskConfig)
    s3: S3Config = field(default_factory=S3Config)
    record: bool = True
    record_depth: int = 1

    def _to_java(self):
        cls = jpype.JClass("group.gnometrading.backtest.config.BacktestConfig")
        ArrayList = jpype.JClass("java.util.ArrayList")
        LinkedHashMap = jpype.JClass("java.util.LinkedHashMap")
        LocalDateTime = jpype.JClass("java.time.LocalDateTime")

        start = self.start_date if isinstance(self.start_date, datetime) else datetime(
            self.start_date.year, self.start_date.month, self.start_date.day
        )
        end = self.end_date if isinstance(self.end_date, datetime) else datetime(
            self.end_date.year, self.end_date.month, self.end_date.day
        )

        obj = cls()
        obj.startDate = LocalDateTime.of(
            start.year, start.month, start.day, start.hour, start.minute, start.second
        )
        obj.endDate = LocalDateTime.of(
            end.year, end.month, end.day, end.hour, end.minute, end.second
        )

        listings_list = ArrayList()
        for listing in self.listings:
            listings_list.add(listing._to_java())
        obj.listings = listings_list

        profiles_map = LinkedHashMap()
        for name, profile in self.profiles.items():
            profiles_map.put(str(name), profile._to_java())
        obj.profiles = profiles_map

        if self.strategy is not None:
            obj.strategy = self.strategy._to_java()
        obj.risk = self.risk._to_java()
        obj.s3 = self.s3._to_java()
        obj.record = self.record
        obj.recordDepth = jpype.JInt(self.record_depth)

        return obj
