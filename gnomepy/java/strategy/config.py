from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import yaml


@dataclass
class StaticFeeConfig:
    taker_fee: float = 0.0
    maker_fee: float = 0.0

    def to_properties(self, prefix: str) -> dict[str, str]:
        return {
            f"{prefix}.model": "static",
            f"{prefix}.taker": str(self.taker_fee),
            f"{prefix}.maker": str(self.maker_fee),
        }


@dataclass
class ParametricFeeConfig:
    taker_fee_rate: float = 0.07
    maker_fee_rate: float = 0.0

    def to_properties(self, prefix: str) -> dict[str, str]:
        return {
            f"{prefix}.model": "parametric",
            f"{prefix}.taker.rate": str(self.taker_fee_rate),
            f"{prefix}.maker.rate": str(self.maker_fee_rate),
        }


@dataclass
class StaticLatencyConfig:
    latency_nanos: int = 0

    def to_properties(self, prefix: str) -> dict[str, str]:
        return {
            f"{prefix}.model": "static",
            f"{prefix}.nanos": str(self.latency_nanos),
        }


@dataclass
class GaussianLatencyConfig:
    mu: float = 0.0
    sigma: float = 0.0

    def to_properties(self, prefix: str) -> dict[str, str]:
        return {
            f"{prefix}.model": "gaussian",
            f"{prefix}.mu": str(self.mu),
            f"{prefix}.sigma": str(self.sigma),
        }


@dataclass
class MakerTakerLatencyConfig:
    base_nanos: int = 0
    taker_delay_nanos: int = 0
    maker_delay_nanos: int = 0

    def to_properties(self, prefix: str) -> dict[str, str]:
        return {
            f"{prefix}.model": "maker_taker",
            f"{prefix}.base.nanos": str(self.base_nanos),
            f"{prefix}.taker.delay.nanos": str(self.taker_delay_nanos),
            f"{prefix}.maker.delay.nanos": str(self.maker_delay_nanos),
        }


@dataclass
class OptimisticQueueConfig:
    def to_properties(self, prefix: str) -> dict[str, str]:
        return {f"{prefix}.model": "optimistic"}


@dataclass
class RiskAverseQueueConfig:
    def to_properties(self, prefix: str) -> dict[str, str]:
        return {f"{prefix}.model": "risk_averse"}


@dataclass
class ProbabilisticQueueConfig:
    cancel_ahead_probability: float = 0.5

    def to_properties(self, prefix: str) -> dict[str, str]:
        return {
            f"{prefix}.model": "probabilistic",
            f"{prefix}.cancel.ahead.probability": str(self.cancel_ahead_probability),
        }


FeeConfig = Union[StaticFeeConfig, ParametricFeeConfig]
LatencyModelConfig = Union[StaticLatencyConfig, GaussianLatencyConfig, MakerTakerLatencyConfig]
QueueConfig = Union[OptimisticQueueConfig, RiskAverseQueueConfig, ProbabilisticQueueConfig]


@dataclass
class SimulationProfile:
    fee: FeeConfig = field(default_factory=StaticFeeConfig)
    network_latency: LatencyModelConfig = field(default_factory=StaticLatencyConfig)
    order_latency: LatencyModelConfig = field(default_factory=StaticLatencyConfig)
    queue: QueueConfig = field(default_factory=RiskAverseQueueConfig)

    def to_properties(self, prefix: str) -> dict[str, str]:
        props: dict[str, str] = {}
        props.update(self.fee.to_properties(f"{prefix}.fee"))
        props.update(self.network_latency.to_properties(f"{prefix}.network.latency"))
        props.update(self.order_latency.to_properties(f"{prefix}.order.latency"))
        props.update(self.queue.to_properties(f"{prefix}.queue"))
        return props


@dataclass
class ListingSessionConfig:
    listing_id: int
    profile: str


@dataclass
class StrategyConfig:
    class_name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionConfig:
    mode: str
    listings: list[ListingSessionConfig]
    profiles: dict[str, SimulationProfile]
    session_id: str | None = None
    strategy_id: int | None = None
    strategy: StrategyConfig | None = None

    def to_properties(self) -> dict[str, Any]:
        props: dict[str, Any] = {
            "mode": self.mode,
            "listings": ",".join(str(lsc.listing_id) for lsc in self.listings),
        }
        if self.session_id is not None:
            props["session.id"] = self.session_id
        if self.strategy_id is not None:
            props["strategy.id"] = str(self.strategy_id)
        if self.strategy:
            is_python = ":" in self.strategy.class_name
            props["strategy.type"] = "python" if is_python else "java"
            if not is_python:
                props["strategy.class"] = self.strategy.class_name
        if self.strategy and self.strategy.args:
            for k, v in self.strategy.args.items():
                props[f"strategy.args.{k}"] = v
        if self.mode == "paper":
            for name, profile in self.profiles.items():
                props.update(profile.to_properties(f"simulation.profiles.{name}"))
            for lsc in self.listings:
                props[f"simulation.listing.{lsc.listing_id}.profile"] = lsc.profile
        return props

    @staticmethod
    def from_yaml(path: str | Path) -> SessionConfig:
        data = yaml.safe_load(Path(path).read_text())
        strategy = None
        if "strategy" in data:
            s = data["strategy"]
            strategy = StrategyConfig(
                class_name=s["class_name"],
                args=s.get("args") or {},
            )

        raw_listings = data.get("listings", [])
        listings = [
            ListingSessionConfig(listing_id=entry["listing_id"], profile=entry["profile"])
            for entry in raw_listings
        ]

        raw_profiles = data.get("profiles", {})
        profiles = {name: _parse_simulation_profile(cfg) for name, cfg in raw_profiles.items()}

        if data.get("mode") == "paper":
            for lsc in listings:
                if lsc.profile not in profiles:
                    raise ValueError(
                        f"Listing {lsc.listing_id} references profile '{lsc.profile}' which is not defined"
                    )

        return SessionConfig(
            session_id=data.get("session_id"),
            strategy_id=data.get("strategy_id"),
            mode=data["mode"],
            listings=listings,
            profiles=profiles,
            strategy=strategy,
        )


def _parse_simulation_profile(sim: dict) -> SimulationProfile:
    fee = _parse_fee_config(sim.get("fee", {}))
    network_latency = _parse_latency_config(sim.get("network_latency", {}))
    order_latency = _parse_latency_config(sim.get("order_latency", {}))
    queue = _parse_queue_config(sim.get("queue", {}))
    return SimulationProfile(fee=fee, network_latency=network_latency, order_latency=order_latency, queue=queue)


def _parse_fee_config(cfg: dict) -> FeeConfig:
    model_type = cfg.get("model", "static")
    if model_type == "parametric":
        return ParametricFeeConfig(
            taker_fee_rate=cfg.get("taker_fee_rate", 0.07),
            maker_fee_rate=cfg.get("maker_fee_rate", 0.0),
        )
    return StaticFeeConfig(taker_fee=cfg.get("taker_fee", 0.0), maker_fee=cfg.get("maker_fee", 0.0))


def _parse_latency_config(cfg: dict) -> LatencyModelConfig:
    model_type = cfg.get("model", "static")
    if model_type == "gaussian":
        return GaussianLatencyConfig(mu=cfg.get("mu", 0.0), sigma=cfg.get("sigma", 0.0))
    if model_type == "maker_taker":
        return MakerTakerLatencyConfig(
            base_nanos=cfg.get("base_nanos", 0),
            taker_delay_nanos=cfg.get("taker_delay_nanos", 0),
            maker_delay_nanos=cfg.get("maker_delay_nanos", 0),
        )
    return StaticLatencyConfig(latency_nanos=cfg.get("latency_nanos", 0))


def _parse_queue_config(cfg: dict) -> QueueConfig:
    model_type = cfg.get("model", "risk_averse")
    if model_type == "optimistic":
        return OptimisticQueueConfig()
    if model_type == "probabilistic":
        return ProbabilisticQueueConfig(cancel_ahead_probability=cfg.get("cancel_ahead_probability", 0.5))
    return RiskAverseQueueConfig()
