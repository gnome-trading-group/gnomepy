"""Public config parsing for YAML backtest configs.

Used by both the CLI (``gnomepy backtest --config``) and
``gnomepy_research.presets``.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from gnomepy.java.backtest.config import (
    ExchangeConfig,
    GaussianLatencyConfig,
    OptimisticQueueConfig,
    ProbabilisticQueueConfig,
    RiskAverseQueueConfig,
    StaticFeeConfig,
    StaticLatencyConfig,
)
from gnomepy.java.backtest.strategy import Strategy
from gnomepy.java.enums import SchemaType


def load_strategy(import_path: str, kwargs: dict[str, Any] | None = None) -> Strategy:
    """Resolve a ``'pkg.module:ClassName'`` import path and instantiate it.

    Raises ``ValueError`` if the path format is wrong or the result is not
    a ``Strategy``.
    """
    if ":" not in import_path:
        raise ValueError(f"strategy must be 'module.path:ClassName', got: {import_path!r}")
    module_path, class_name = import_path.split(":", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    instance = cls(**(kwargs or {}))
    if not isinstance(instance, Strategy):
        raise ValueError(
            f"{import_path} did not produce a gnomepy.Strategy instance "
            f"(got {type(instance).__name__})"
        )
    return instance


def parse_schema(s: str) -> SchemaType:
    """Parse a schema type name (e.g. ``'MBP_10'``) into a ``SchemaType`` enum."""
    try:
        return SchemaType[s]
    except KeyError as e:
        valid = ", ".join(m.name for m in SchemaType)
        raise ValueError(f"unknown schema {s!r}; valid: {valid}") from e


def exchange_from_dict(d: dict[str, Any]) -> ExchangeConfig:
    """Build an ``ExchangeConfig`` from a flat or nested dict."""
    kwargs: dict[str, Any] = {
        "exchange_id": int(d["exchange_id"]),
        "security_id": int(d["security_id"]),
    }
    if "fee" in d:
        kwargs["fee_model"] = StaticFeeConfig(**d["fee"])
    if "network_latency_ns" in d:
        kwargs["network_latency"] = StaticLatencyConfig(int(d["network_latency_ns"]))
    elif "network_latency" in d:
        kwargs["network_latency"] = _latency_from_dict(d["network_latency"])
    if "order_processing_latency_ns" in d:
        kwargs["order_processing_latency"] = StaticLatencyConfig(
            int(d["order_processing_latency_ns"])
        )
    elif "order_processing_latency" in d:
        kwargs["order_processing_latency"] = _latency_from_dict(
            d["order_processing_latency"]
        )
    if "queue" in d:
        kwargs["queue_model"] = _queue_from_dict(d["queue"])
    return ExchangeConfig(**kwargs)


def _latency_from_dict(d: dict[str, Any]):
    if "mu" in d or "sigma" in d:
        return GaussianLatencyConfig(mu=float(d.get("mu", 0)), sigma=float(d.get("sigma", 0)))
    return StaticLatencyConfig(int(d.get("latency_nanos", 0)))


def _queue_from_dict(d: dict[str, Any]):
    kind = d.get("kind", "risk_averse").lower()
    if kind == "optimistic":
        return OptimisticQueueConfig()
    if kind == "risk_averse":
        return RiskAverseQueueConfig()
    if kind == "probabilistic":
        return ProbabilisticQueueConfig(
            cancel_ahead_probability=float(d.get("cancel_ahead_probability", 0.5))
        )
    raise ValueError(f"unknown queue kind {kind!r}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def backtest_from_dict(config: dict[str, Any]) -> "Backtest":
    """Build a ready-to-run ``Backtest`` from a config dict (as loaded from YAML).

    Returns the ``Backtest`` instance — call ``.run()`` to execute it.
    """
    from gnomepy.java.backtest.runner import Backtest

    strategy_args = config.get("strategy_args") or {}

    if "java_strategy" in config:
        strategy = config["java_strategy"]
        java_strategy_args = strategy_args
    elif "strategy" in config:
        strategy = load_strategy(config["strategy"], strategy_args)
        java_strategy_args = None
    else:
        raise ValueError("config must define either 'strategy' or 'java_strategy'")

    return Backtest(
        strategy=strategy,
        schema_type=parse_schema(config["schema_type"]),
        start_date=config["start_date"],
        end_date=config["end_date"],
        exchanges=[exchange_from_dict(e) for e in config["exchanges"]],
        bucket=config.get("bucket", "gnome-market-data-prod"),
        extra_jars=list(config.get("jars") or []) or None,
        strategy_args=java_strategy_args,
    )
