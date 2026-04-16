"""Command-line entrypoint for launching backtests.

Usage:
    gnomepy backtest --config backtest.yaml \\
        [--strategy mymodule:MyStrategy | --java-strategy com.example.MyStrategy] \\
        [--output PATH | --s3-bucket bucket-name] \\
        [--job-id ID] \\
        [--no-progress]
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import click
from importlib.metadata import version as _pkg_version

from gnomepy.java.backtest.runner import run_backtest
from gnomepy.utils import uuid7

# Backward-compatible alias kept for external code that imports _uuid7 from this module.
_uuid7 = uuid7


@click.group()
def main() -> None:
    """gnomepy backtest CLI."""


@main.command(name="backtest")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="YAML backtest config file",
)
@click.option(
    "--strategy",
    "strategy_path",
    help="Python strategy override as module.path:ClassName",
)
@click.option(
    "--java-strategy",
    "java_strategy",
    help="Java strategy override as fully-qualified class name",
)
@click.option("--output", type=click.Path(), help="Directory to save BacktestResults")
@click.option("--s3-bucket", "s3_bucket", help="S3 bucket to persist results (saves to s3://<bucket>/backtests/<job_id>)")
@click.option("--job-id", "job_id", help="Job id (default: generated UUIDv7)")
@click.option("--no-progress", is_flag=True, help="Disable progress output")
def backtest(
    config: str,
    strategy_path: str | None,
    java_strategy: str | None,
    output: str | None,
    s3_bucket: str | None,
    job_id: str | None,
    no_progress: bool,
) -> None:
    """Run a backtest from a YAML config file."""
    if strategy_path and java_strategy:
        raise click.UsageError("pass either --strategy or --java-strategy, not both")

    job_id = job_id or uuid7()
    click.echo(f"job_id: {job_id}")

    strategy = strategy_path or java_strategy or None

    results = run_backtest(
        config,
        strategy=strategy,
        backtest_id=job_id,
        progress=not no_progress,
    )

    if results is not None:
        # Enrich metadata with CLI-specific provenance before persisting.
        if results.metadata is not None:
            results.metadata.config_path = str(config)
            try:
                results.metadata.gnomepy_research_version = _pkg_version("gnomepy_research")
            except Exception:
                pass
            results.metadata.gnomepy_research_commit = os.environ.get("RESEARCH_COMMIT")

        if s3_bucket is not None:
            s3_path = f"s3://{s3_bucket}/backtests/{job_id}"
            results.save(s3_path)
            click.echo(f"\nresults saved to {s3_path}")
        else:
            out_dir = Path(output) if output else Path.cwd() / job_id
            results.save(out_dir)
            shutil.copyfile(config, out_dir / "config.yaml")

            # Write manifest.json for backward compatibility with tooling that reads it.
            meta = results.metadata
            manifest = {
                "job_id": job_id,
                "created_at": meta.created_at if meta else datetime.now(timezone.utc).isoformat(),
                "gnomepy_version": meta.gnomepy_version if meta else None,
                "gnomepy_research_version": meta.gnomepy_research_version if meta else None,
                "gnomepy_research_commit": meta.gnomepy_research_commit if meta else None,
                "config": "config.yaml",
            }
            (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            click.echo(f"\nresults saved to {out_dir}")


if __name__ == "__main__":
    main()
