from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SimulationConfig:
    taker_fee: float = 0.0
    maker_fee: float = 0.0
    network_latency_nanos: int = 0
    order_latency_nanos: int = 0
    queue_model: str = "risk_averse"
    cancel_ahead_probability: float = 0.5


@dataclass
class StrategyConfig:
    class_name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionConfig:
    mode: str
    listings: list[int]
    strategy_id: int | None = None
    strategy: StrategyConfig | None = None
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    def to_properties(self) -> dict[str, str]:
        """Flatten to the Properties key format expected by TradingOrchestrator."""
        props: dict[str, str] = {
            "mode": self.mode,
            "listings": ",".join(str(lid) for lid in self.listings),
        }
        if self.strategy_id is not None:
            props["strategy.id"] = str(self.strategy_id)
        if self.strategy:
            is_python = ":" in self.strategy.class_name
            props["strategy.type"] = "python" if is_python else "java"
            if not is_python:
                props["strategy.class"] = self.strategy.class_name
        if self.strategy and self.strategy.args:
            for k, v in self.strategy.args.items():
                props[f"strategy.args.{k}"] = str(v)
        if self.mode == "paper":
            sim = self.simulation
            props["simulation.taker.fee"] = str(sim.taker_fee)
            props["simulation.maker.fee"] = str(sim.maker_fee)
            props["simulation.network.latency.nanos"] = str(sim.network_latency_nanos)
            props["simulation.order.latency.nanos"] = str(sim.order_latency_nanos)
            props["simulation.queue.model"] = sim.queue_model
            props["simulation.queue.cancel.ahead.probability"] = str(sim.cancel_ahead_probability)
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
        simulation = SimulationConfig()
        if "simulation" in data:
            sim = data["simulation"]
            simulation = SimulationConfig(
                taker_fee=sim.get("taker_fee", 0.0),
                maker_fee=sim.get("maker_fee", 0.0),
                network_latency_nanos=sim.get("network_latency_nanos", 0),
                order_latency_nanos=sim.get("order_latency_nanos", 0),
                queue_model=sim.get("queue_model", "risk_averse"),
                cancel_ahead_probability=sim.get("cancel_ahead_probability", 0.5),
            )
        return SessionConfig(
            strategy_id=data.get("strategy_id"),
            mode=data["mode"],
            listings=list(data["listings"]),
            strategy=strategy,
            simulation=simulation,
        )
