"""Command-line interface for gnomepy."""
from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import click
import yaml
from importlib.metadata import version as _pkg_version

from gnomepy.auth import get_id_token as _get_id_token
from gnomepy.auth import login as _auth_login
from gnomepy.auth import logout as _auth_logout
from gnomepy.java.backtest.runner import run_backtest
from gnomepy.remote import cancel_backtest, get_backtest, list_backtests, submit_backtest
from gnomepy.sweep import expand_sweep, sweep_params
from gnomepy.utils import uuid7


@click.group()
def main() -> None:
    """gnomepy — backtesting infrastructure for the gnome trading system."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

@main.command()
def login() -> None:
    """Authenticate with the gnome controller (opens browser for SSO)."""
    click.echo("Opening browser for authentication...")
    try:
        _auth_login()
        click.echo("Login successful. Credentials saved to ~/.gnomepy/credentials.json")
    except RuntimeError as e:
        raise click.ClickException(str(e))


@main.command()
def logout() -> None:
    """Remove cached authentication credentials."""
    _auth_logout()
    click.echo("Logged out.")


# ---------------------------------------------------------------------------
# Backtest commands
# ---------------------------------------------------------------------------

@main.group()
def backtest() -> None:
    """Run and manage backtests."""


@backtest.command(name="run")
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
@click.option(
    "--output",
    help="Directory or s3:// URI to save results (default: ./<job_id>)",
)
@click.option("--job-id", "job_id", help="Job ID (default: generated UUIDv7)")
@click.option("--no-progress", is_flag=True, help="Disable progress output")
def backtest_run(
    config: str,
    strategy_path: str | None,
    java_strategy: str | None,
    output: str | None,
    job_id: str | None,
    no_progress: bool,
) -> None:
    """Run a backtest locally and generate a report."""
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

    if results is None:
        return

    if results.metadata and results.metadata.warnings:
        for w in results.metadata.warnings:
            click.echo(f"warning: {w}", err=True)

    if results.metadata is not None:
        results.metadata.config_path = str(config)
        try:
            results.metadata.gnomepy_research_version = _pkg_version("gnomepy_research")
        except Exception:
            pass
        results.metadata.gnomepy_research_commit = os.environ.get("RESEARCH_COMMIT")

    if output is not None and output.startswith("s3://"):
        results.save(output)
        click.echo(f"results saved to {output}")
        _generate_report(results, output)
    else:
        out_dir = Path(output) if output else Path.cwd() / job_id
        results.save(out_dir)
        shutil.copyfile(config, out_dir / "config.yaml")
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
        click.echo(f"results saved to {out_dir}")
        _generate_report(results, str(out_dir))


def _generate_report(results, base_path: str) -> None:
    """Generate report.html and summary.json alongside the backtest results."""
    try:
        from gnomepy.reporting import BacktestReport
        report = BacktestReport(results)

        if base_path.startswith("s3://"):
            report.save_html(f"{base_path}/report.html")
            report.save_summary(f"{base_path}/summary.json")
            click.echo(f"report saved to {base_path}/report.html")
        else:
            out = Path(base_path)
            report.save_html(out / "report.html")
            report.save_summary(out / "summary.json")
            click.echo(f"report saved to {out / 'report.html'}")
    except Exception as e:
        click.echo(f"warning: report generation failed: {e}", err=True)


# ---------------------------------------------------------------------------
# Remote backtest commands (requires API + AWS infrastructure)
# ---------------------------------------------------------------------------

@backtest.command(name="submit")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="YAML backtest config file (supports sweep syntax)",
)
@click.option("--research-commit", default="main", show_default=True, help="gnomepy-research git ref")
@click.option("--dry-run", is_flag=True, help="Preview sweep expansion without submitting")
def backtest_submit(config: str, research_commit: str, dry_run: bool) -> None:
    """Submit a backtest (or parameter sweep) to AWS Batch."""
    config_yaml = Path(config).read_text()
    parsed = yaml.safe_load(config_yaml)
    params = sweep_params(parsed)

    if params:
        configs = expand_sweep(parsed)
        click.echo(f"sweep: {len(configs)} jobs across {list(params.keys())}")
        for param, values in params.items():
            click.echo(f"  {param}: {values}")
    else:
        click.echo("no sweep parameters — single job")

    if dry_run:
        return

    result = submit_backtest(config_yaml, research_commit=research_commit)
    click.echo(f"run_id:    {result['run_id']}")
    click.echo(f"job_count: {result['job_count']}")
    click.echo(f"status:    {result['status']}")
    click.echo(f"batch_job: {result['batch_job_id']}")


@backtest.command(name="status")
@click.argument("run_id")
@click.option("--jobs", is_flag=True, help="Show individual job statuses")
def backtest_status(run_id: str, jobs: bool) -> None:
    """Check the status of a submitted backtest run."""
    run = get_backtest(run_id)
    click.echo(f"run_id:    {run['run_id']}")
    click.echo(f"status:    {run['status']}")
    click.echo(f"strategy:  {run.get('strategy', 'unknown')}")
    click.echo(f"jobs:      {run.get('completed_count', 0)}/{run.get('job_count', '?')} completed, {run.get('failed_count', 0)} failed")
    click.echo(f"submitted: {run.get('submitted_at', '')}")

    if jobs:
        job_list = run.get("jobs", [])
        if not job_list:
            click.echo("  (no job records yet)")
        for job in job_list:
            idx = job.get("array_index", 0)
            st = job.get("status", "?")
            pnl = job.get("final_pnl")
            sharpe = job.get("sharpe")
            params_str = ", ".join(f"{k}={v}" for k, v in job.get("config_params", {}).items())
            line = f"  [{idx:04d}] {st:<12} {params_str}"
            if pnl is not None:
                line += f"  pnl={pnl:.4f}"
            if sharpe is not None:
                line += f"  sharpe={sharpe:.3f}"
            if job.get("report_url"):
                line += "  [report available]"
            click.echo(line)


@backtest.command(name="list")
@click.option("--status", help="Filter by status (e.g. RUNNING, COMPLETED)")
@click.option("--limit", default=20, show_default=True, help="Number of results")
def backtest_list(status: str | None, limit: int) -> None:
    """List recent backtest runs."""
    result = list_backtests(status=status, limit=limit)
    runs = result.get("runs", [])
    if not runs:
        click.echo("no backtest runs found")
        return
    header = f"{'RUN_ID':<20} {'STATUS':<18} {'STRATEGY':<35} {'JOBS':>5} {'SUBMITTED'}"
    click.echo(header)
    click.echo("-" * len(header))
    for run in runs:
        run_id = run.get("run_id", "")[-12:]  # last 12 chars of hex timestamp
        status_val = run.get("status", "")
        strategy = run.get("strategy", "")[-35:]
        jobs = f"{run.get('completed_count', 0)}/{run.get('job_count', '?')}"
        submitted = run.get("submitted_at", "")[:19].replace("T", " ")
        click.echo(f"{run_id:<20} {status_val:<18} {strategy:<35} {jobs:>5} {submitted}")


@backtest.command(name="results")
@click.argument("run_id")
def backtest_results(run_id: str) -> None:
    """Show report URLs for a completed backtest run."""
    run = get_backtest(run_id)
    jobs = run.get("jobs", [])
    has_reports = any(j.get("report_url") for j in jobs)
    if not has_reports:
        click.echo("no reports available (run may still be in progress)")
        return
    for job in jobs:
        url = job.get("report_url")
        if url:
            idx = job.get("array_index", 0)
            params_str = ", ".join(f"{k}={v}" for k, v in job.get("config_params", {}).items())
            click.echo(f"[{idx:04d}] {params_str}")
            click.echo(f"       {url}")


@backtest.command(name="cancel")
@click.argument("run_id")
@click.confirmation_option(prompt="Cancel this run?")
def backtest_cancel(run_id: str) -> None:
    """Cancel a running backtest."""
    result = cancel_backtest(run_id)
    click.echo(f"run_id: {result['run_id']}")
    click.echo(f"status: {result['status']}")


if __name__ == "__main__":
    main()
