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

import json
import os
import secrets
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

from gnomepy.backtest_config import (
    exchange_from_dict as _exchange_from_dict,
    load_strategy as _load_strategy,
    load_yaml as _load_yaml_file,
    parse_schema as _parse_schema,
)
from gnomepy.entrypoint import run_backtest


# ---------------------------------------------------------------------------
# CLI-only helpers (KV pair parsing for --strategy-arg / --exchange flags)
# ---------------------------------------------------------------------------

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


def _uuid7() -> str:
    """Generate a UUIDv7 (time-ordered) as a hex string with dashes.

    Layout: 48-bit unix-ms timestamp | 4-bit version (7) | 12-bit rand_a
            | 2-bit variant (10) | 62-bit rand_b
    """
    ms = int(time.time() * 1000) & 0xFFFFFFFFFFFF
    rand_a = secrets.randbits(12)
    rand_b = secrets.randbits(62)
    n = (ms << 80) | (0x7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b
    h = f"{n:032x}"
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _load_yaml(path: str) -> dict[str, Any]:
    return _load_yaml_file(path)


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
@click.option("--output", type=click.Path(), help="Directory to save BacktestResults (default: ./<job-id>)")
@click.option("--job-id", "job_id", help="Job id (default: generated UUIDv7)")
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
    job_id: str | None,
    no_progress: bool,
) -> None:
    """Run a backtest."""
    job_id = job_id or _uuid7()
    click.echo(f"job_id: {job_id}")

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

    if results is not None:
        out_dir = Path(output) if output else Path.cwd() / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        results.save(out_dir)

        if config:
            shutil.copyfile(config, out_dir / "config.yaml")

        try:
            from importlib.metadata import version as _pkg_version
            gnomepy_version = _pkg_version("gnomepy")
        except Exception:
            gnomepy_version = None
        try:
            gnomepy_research_version = _pkg_version("gnomepy_research")
        except Exception:
            gnomepy_research_version = None

        manifest = {
            "job_id": job_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "gnomepy_version": gnomepy_version,
            "gnomepy_research_version": gnomepy_research_version,
            "gnomepy_research_commit": os.environ.get("RESEARCH_COMMIT"),
            "schema_type": schema_type.name,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "bucket": bucket,
            "config": "config.yaml" if config else None,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

        click.echo(f"\nresults saved to {out_dir}")


if __name__ == "__main__":
    main()
