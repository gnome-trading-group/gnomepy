"""Command-line entrypoint for launching backtests.

Two modes:

  Mode A — import path:
      gnomepy run --strategy my_pkg.mod:MyStrategy \\
                  --schema MBP_10 \\
                  --start 2026-01-15 --end 2026-01-16 \\
                  --exchange exchange_id=1,security_id=100 \\
                  [--strategy-arg key=value ...] \\
                  [--bucket ...] [--output PATH] [--no-progress]

  Mode B — YAML config:
      gnomepy run --config backtest.yaml [--output PATH] [--no-progress]
"""
from __future__ import annotations

import importlib
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from gnomepy.entrypoint import run_backtest
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


# ---------------------------------------------------------------------------
# Strategy + parsing helpers
# ---------------------------------------------------------------------------

def _load_strategy(import_path: str, kwargs: dict[str, Any]) -> Strategy:
    """Resolve a 'pkg.module:ClassName' import path and instantiate it."""
    if ":" not in import_path:
        raise click.BadParameter(
            f"--strategy must be 'module.path:ClassName', got: {import_path!r}"
        )
    module_path, class_name = import_path.split(":", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    instance = cls(**kwargs)
    if not isinstance(instance, Strategy):
        raise click.BadParameter(
            f"{import_path} did not produce a gnomepy.Strategy instance "
            f"(got {type(instance).__name__})"
        )
    return instance


def _parse_kv_pairs(items: tuple[str, ...] | list[str]) -> dict[str, Any]:
    """Parse 'key=value' pairs into a dict, with simple int/float/bool coercion."""
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise click.BadParameter(f"expected key=value, got: {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = _coerce(v.strip())
    return out


def _coerce(v: str) -> Any:
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _parse_schema(s: str) -> SchemaType:
    try:
        return SchemaType[s]
    except KeyError as e:
        valid = ", ".join(m.name for m in SchemaType)
        raise click.BadParameter(f"unknown schema {s!r}; valid: {valid}") from e


def _exchange_from_dict(d: dict[str, Any]) -> ExchangeConfig:
    """Build an ExchangeConfig from a flat or nested dict."""
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
    raise click.BadParameter(f"unknown queue kind {kind!r}")


def _load_yaml(path: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise click.ClickException("PyYAML is required for --config; pip install pyyaml") from e
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def main() -> None:
    """gnomepy backtest CLI."""


@main.command(name="backtest")
@click.option("--config", type=click.Path(exists=True, dir_okay=False), help="YAML config file")
@click.option("--strategy", "strategy_path", help="Python strategy as module.path:ClassName")
@click.option(
    "--java-strategy",
    "java_strategy",
    help="Java strategy as fully-qualified class name (e.g. com.example.MyStrategy)",
)
@click.option(
    "--strategy-arg",
    "strategy_args",
    multiple=True,
    help="Strategy kwarg as key=value (repeatable). "
    "Python: passed to __init__. Java: passed via configure(Map).",
)
@click.option(
    "--jar",
    "jars",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Extra JAR to add to the JVM classpath (repeatable). "
    "Use for Java strategies in JARs not already discovered.",
)
@click.option("--schema", help="SchemaType name (e.g. MBP_10)")
@click.option(
    "--start",
    type=click.DateTime(formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"]),
    help="Start date (YYYY-MM-DD or ISO datetime)",
)
@click.option(
    "--end",
    type=click.DateTime(formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"]),
    help="End date (YYYY-MM-DD or ISO datetime)",
)
@click.option(
    "--exchange",
    "exchanges",
    multiple=True,
    help="Exchange config as comma-separated key=value (repeatable). "
    "Example: --exchange exchange_id=1,security_id=100",
)
@click.option("--bucket", default="gnome-market-data-prod", show_default=True)
@click.option("--output", type=click.Path(), help="Directory to save BacktestResults")
@click.option("--no-progress", is_flag=True, help="Disable progress output")
def backtest(
    config: str | None,
    strategy_path: str | None,
    java_strategy: str | None,
    strategy_args: tuple[str, ...],
    jars: tuple[str, ...],
    schema: str | None,
    start: datetime | None,
    end: datetime | None,
    exchanges: tuple[str, ...],
    bucket: str,
    output: str | None,
    no_progress: bool,
) -> None:
    """Run a backtest."""
    java_strategy_args: dict | None = None
    extra_jars: list[str] | None = None

    if config:
        cfg = _load_yaml(config)
        cfg_args = cfg.get("strategy_args") or {}
        if "java_strategy" in cfg:
            strategy = cfg["java_strategy"]  # FQN string passed straight to runner
            java_strategy_args = cfg_args
        elif "strategy" in cfg:
            strategy = _load_strategy(cfg["strategy"], cfg_args)
        else:
            raise click.UsageError("config must define either 'strategy' or 'java_strategy'")
        schema_type = _parse_schema(cfg["schema_type"])
        start_date = cfg["start_date"]
        end_date = cfg["end_date"]
        exchange_configs = [_exchange_from_dict(e) for e in cfg["exchanges"]]
        bucket = cfg.get("bucket", bucket)
        extra_jars = list(cfg.get("jars") or []) or None
    else:
        if strategy_path and java_strategy:
            raise click.UsageError("pass either --strategy or --java-strategy, not both")
        if not (strategy_path or java_strategy):
            raise click.UsageError("either --config, --strategy, or --java-strategy is required")
        if not (schema and start and end):
            raise click.UsageError("--schema, --start, --end are required without --config")
        if not exchanges:
            raise click.UsageError("at least one --exchange is required")

        parsed_args = _parse_kv_pairs(strategy_args)
        if java_strategy:
            strategy = java_strategy
            java_strategy_args = parsed_args
        else:
            strategy = _load_strategy(strategy_path, parsed_args)

        schema_type = _parse_schema(schema)
        start_date = start
        end_date = end
        exchange_configs = [
            _exchange_from_dict(_parse_kv_pairs(e.split(","))) for e in exchanges
        ]
        extra_jars = list(jars) or None

    results = run_backtest(
        strategy=strategy,
        schema_type=schema_type,
        start_date=start_date,
        end_date=end_date,
        exchanges=exchange_configs,
        bucket=bucket,
        progress=not no_progress,
        strategy_args=java_strategy_args,
        extra_jars=extra_jars,
    )

    if output and results is not None:
        results.save(Path(output))
        click.echo(f"\nresults saved to {output}")


if __name__ == "__main__":
    main()
