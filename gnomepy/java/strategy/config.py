from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# The blocks a run may set in place of its exchange's registry simulation parameters, in the shape backtest configs
# take (each a model named by `type`, or a self-trade prevention mode).
SIMULATION_BLOCKS = (
    "fee_model",
    "network_latency",
    "order_processing_latency",
    "market_data_latency",
    "queue_model",
    "self_trade_prevention",
)


def _overrides(where: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping of simulation blocks")
    unknown = sorted(set(value) - set(SIMULATION_BLOCKS))
    if unknown:
        raise ValueError(f"{where} has unknown blocks: {', '.join(unknown)}")
    return value


# Stream ids double as Aeron stream ids, which are int32.
_MAX_STREAM_ID = 2**31 - 1


def _streams(value: Any) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list) or any(
        isinstance(v, bool) or not isinstance(v, int) or not 0 < v <= _MAX_STREAM_ID for v in value
    ):
        raise ValueError("streams must be a list of positive int32 stream ids")
    return list(dict.fromkeys(value))


@dataclass
class ListingSessionConfig:
    listing_id: int
    # Blocks this listing simulates with in place of its exchange's.
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    class_name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionConfig:
    mode: str
    listings: list[ListingSessionConfig]
    # Blocks every listing on an exchange (by code, e.g. KALSHI) simulates with in place of the exchange's registry
    # parameters. Listings simulate as their exchange is recorded otherwise.
    exchanges: dict[str, dict[str, Any]] = field(default_factory=dict)
    session_id: str | None = None
    strategy_id: int | None = None
    strategy: StrategyConfig | None = None
    seed: int | None = None
    # The time the session reads the registry's simulation parameters and fees as known at (ISO-8601). A deployed
    # session is pinned by the registry when it launches; a local run pins when it starts.
    simulation_as_of: str | None = None
    # Market-data stream ids (registry md.stream) the session reads without trading them, from any region; a traded
    # listing's own book stream is implied. Paper only: live sessions still open their own feeds, and the launcher
    # refuses a live session that sets this.
    streams: list[int] = field(default_factory=list)

    def simulation(self) -> dict[str, Any]:
        """The session's ``simulation`` config value, as the orchestrator reads it."""
        simulation: dict[str, Any] = {}
        if self.simulation_as_of is not None:
            simulation["as_of"] = self.simulation_as_of
        if self.seed is not None:
            simulation["seed"] = self.seed
        if self.exchanges:
            simulation["exchanges"] = self.exchanges
        listings = {str(lsc.listing_id): lsc.overrides for lsc in self.listings if lsc.overrides}
        if listings:
            simulation["listings"] = listings
        return simulation

    def to_session_config(self) -> dict[str, Any]:
        """The config the registry launches a session with."""
        config: dict[str, Any] = {
            "mode": self.mode,
            "listings": [lsc.listing_id for lsc in self.listings],
        }
        if self.streams:
            config["streams"] = _streams(self.streams)
        if self.session_id is not None:
            config["session.id"] = self.session_id
        if self.strategy_id is not None:
            config["strategy.id"] = str(self.strategy_id)
        if self.strategy:
            is_python = ":" in self.strategy.class_name
            config["strategy.type"] = "python" if is_python else "java"
            config["strategy.class"] = self.strategy.class_name
            for k, v in self.strategy.args.items():
                config[f"strategy.args.{k}"] = v
        if self.mode == "paper":
            config["simulation"] = self.simulation()
        return config

    def to_properties(self) -> dict[str, Any]:
        """Properties for a local run: the session config, with the simulation as the JSON an instance receives."""
        props = self.to_session_config()
        # A local Python strategy is handed to the JVM directly (run_strategy_session), not loaded by class name.
        if props.get("strategy.type") == "python":
            del props["strategy.class"]
        simulation = props.pop("simulation", None)
        if simulation is not None:
            simulation.setdefault("as_of", datetime.now(timezone.utc).isoformat())
            props["simulation.json"] = json.dumps(simulation)
        return props

    @staticmethod
    def from_yaml(path: str | Path) -> SessionConfig:
        data = yaml.safe_load(Path(path).read_text())
        strategy = None
        if "strategy" in data:
            s = data["strategy"]
            strategy = StrategyConfig(class_name=s["class_name"], args=s.get("args") or {})

        listings = [
            ListingSessionConfig(
                listing_id=entry["listing_id"],
                overrides=_overrides(f"listing {entry['listing_id']} overrides", entry.get("overrides")),
            )
            for entry in data.get("listings", [])
        ]
        exchanges = {
            code: _overrides(f"exchanges.{code}", overrides) for code, overrides in (data.get("exchanges") or {}).items()
        }
        return SessionConfig(
            mode=data["mode"],
            listings=listings,
            exchanges=exchanges,
            session_id=data.get("session_id"),
            strategy_id=data.get("strategy_id"),
            strategy=strategy,
            seed=data.get("seed"),
            simulation_as_of=data.get("simulation_as_of"),
            streams=_streams(data.get("streams")),
        )
