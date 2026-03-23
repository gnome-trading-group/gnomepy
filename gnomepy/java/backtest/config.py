from __future__ import annotations

from dataclasses import dataclass, field

import jpype


@dataclass
class StaticFeeConfig:
    taker_fee: float = 0.0
    maker_fee: float = 0.0

    def _to_java(self):
        cls = jpype.JClass("group.gnometrading.backtest.fee.StaticFeeModel")
        return cls(float(self.taker_fee), float(self.maker_fee))


@dataclass
class StaticLatencyConfig:
    latency_nanos: int = 0

    def _to_java(self):
        cls = jpype.JClass("group.gnometrading.backtest.latency.StaticLatency")
        return cls(jpype.JLong(self.latency_nanos))


@dataclass
class GaussianLatencyConfig:
    mu: float = 0.0
    sigma: float = 0.0

    def _to_java(self):
        cls = jpype.JClass("group.gnometrading.backtest.latency.GaussianLatency")
        return cls(float(self.mu), float(self.sigma))


@dataclass
class OptimisticQueueConfig:
    def _to_java(self):
        cls = jpype.JClass(
            "group.gnometrading.backtest.queues.OptimisticQueueModel"
        )
        return cls()


@dataclass
class RiskAverseQueueConfig:
    def _to_java(self):
        cls = jpype.JClass(
            "group.gnometrading.backtest.queues.RiskAverseQueueModel"
        )
        return cls()


@dataclass
class ProbabilisticQueueConfig:
    cancel_ahead_probability: float = 0.5

    def _to_java(self):
        cls = jpype.JClass(
            "group.gnometrading.backtest.queues.ProbabilisticQueueModel"
        )
        return cls(float(self.cancel_ahead_probability))


@dataclass
class ExchangeConfig:
    """Configuration for a simulated exchange instance."""

    exchange_id: int
    security_id: int
    fee_model: StaticFeeConfig = field(default_factory=StaticFeeConfig)
    network_latency: StaticLatencyConfig | GaussianLatencyConfig = field(
        default_factory=StaticLatencyConfig
    )
    order_processing_latency: StaticLatencyConfig | GaussianLatencyConfig = field(
        default_factory=StaticLatencyConfig
    )
    queue_model: OptimisticQueueConfig | RiskAverseQueueConfig | ProbabilisticQueueConfig = field(
        default_factory=RiskAverseQueueConfig
    )
