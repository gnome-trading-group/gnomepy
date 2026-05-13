"""Command-line interface for gnomepy."""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import click
import yaml

from gnomepy.auth import get_id_token as _get_id_token
from gnomepy.auth import login as _auth_login
from gnomepy.auth import logout as _auth_logout
from gnomepy.java.backtest.runner import run_backtest
from gnomepy.remote import cancel_backtest, get_backtest, list_backtests, submit_backtest
from gnomepy.sweep import expand_sweep, get_param_value, sweep_params
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
# Explorer command
# ---------------------------------------------------------------------------

@main.command()
@click.argument("path", required=False, default=None)
@click.option("--run-id", default=None, metavar="RUN_ID", help="Explore a remote run by run_id (looks up S3 path automatically)")
@click.option("--job", "job_index", default=0, show_default=True, help="Job index within a remote run (used with --run-id)")
@click.option("--compare", default=None, metavar="PATH_OR_RUN_ID", help="Second backtest path or run_id for comparison mode")
@click.option("--compare-job", "compare_job_index", default=0, show_default=True, help="Job index for the comparison run (used when --compare is a run_id)")
@click.option("--port", default=8050, show_default=True, help="Port for the Dash server")
@click.option("--no-browser", is_flag=True, help="Do not auto-open the browser")
@click.option("--debug", is_flag=True, hidden=True)
def explore(
    path: str | None,
    run_id: str | None,
    job_index: int,
    compare: str | None,
    compare_job_index: int,
    port: int,
    no_browser: bool,
    debug: bool,
) -> None:
    """Launch the interactive backtest explorer.

    Accepts a local directory, an S3 URI, or a remote run_id:

    \b
      gnomepy explore ./019e2174-...
      gnomepy explore s3://gnome-research-prod/backtests/<run_id>/jobs/0
      gnomepy explore --run-id <run_id> [--job 2]
      gnomepy explore --run-id <run_id> --compare <run_id_b>
    """
    from gnomepy.java.recorder import BacktestResults
    from gnomepy.explorer import launch_explorer

    if path is None and run_id is None:
        raise click.UsageError("Provide either PATH or --run-id.")

    path_a = path if path else _s3_path_for_run(run_id, job_index)
    results_a = BacktestResults.from_parquet(path_a)

    results_b = None
    if compare is not None:
        path_b = compare if _is_local_or_s3(compare) else _s3_path_for_run(compare, compare_job_index)
        results_b = BacktestResults.from_parquet(path_b)

    launch_explorer(results_a, results_b=results_b, port=port, open_browser=not no_browser, debug=debug)


def _is_local_or_s3(path: str) -> bool:
    return path.startswith("s3://") or path.startswith("/") or path.startswith(".")


def _s3_path_for_run(run_id: str, job_index: int) -> str:
    """Resolve the S3 parquet prefix for a remote run from the API."""
    from gnomepy.config import _STAGE
    from gnomepy.remote import get_backtest

    click.echo(f"Looking up run {run_id}…")
    run = get_backtest(run_id)
    status = run.get("status", "")
    if status not in ("COMPLETED", "SUCCEEDED", "PARTIAL"):
        raise click.ClickException(
            f"Run {run_id} has status '{status}' — results may not be available yet."
        )
    jobs = run.get("jobs", [])
    job_count = run.get("job_count", len(jobs))
    if job_index >= job_count:
        raise click.ClickException(
            f"Job index {job_index} out of range — run has {job_count} job(s) (0-{job_count - 1})."
        )
    bucket = f"gnome-research-{_STAGE}"
    s3_path = f"s3://{bucket}/backtests/{run_id}/jobs/{job_index}"
    click.echo(f"Loading {s3_path}")
    return s3_path


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

    strategy = strategy_path or java_strategy or None
    parsed = yaml.safe_load(Path(config).read_text())
    params = sweep_params(parsed)

    if not params:
        _run_single(config, strategy, output, job_id or uuid7(), no_progress)
    else:
        _run_sweep(config, parsed, params, strategy, output, job_id, no_progress)


def _run_single(config: str, strategy, output: str | None, job_id: str, no_progress: bool) -> None:
    click.echo(f"job_id: {job_id}")
    results = run_backtest(config, strategy=strategy, backtest_id=job_id, progress=not no_progress)
    if results is None:
        return

    if results.metadata and results.metadata.warnings:
        for w in results.metadata.warnings:
            click.echo(f"warning: {w}", err=True)

    if output is not None and output.startswith("s3://"):
        results.save(output)
        click.echo(f"results saved to {output}")
        _generate_report(results, output)
    else:
        out_dir = Path(output) if output else Path.cwd() / job_id
        results.save(out_dir)
        shutil.copyfile(config, out_dir / "config.yaml")
        manifest = {"config": "config.yaml"}
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        click.echo(f"results saved to {out_dir}")
        _generate_report(results, str(out_dir))


def _run_sweep(
    config: str,
    parsed: dict,
    params: dict,
    strategy,
    output: str | None,
    sweep_id: str | None,
    no_progress: bool,
) -> None:
    configs = expand_sweep(parsed)
    sweep_id = sweep_id or uuid7()
    click.echo(f"sweep_id: {sweep_id}")
    click.echo(f"sweep: {len(configs)} jobs across {list(params.keys())}")
    for param, values in params.items():
        click.echo(f"  {param}: {values}")

    is_s3 = output is not None and output.startswith("s3://")
    out_base: Path | str = output if is_s3 else (Path(output) if output else Path.cwd() / sweep_id)

    jobs_summary = []
    for i, cfg in enumerate(configs):
        this_job_id = f"{sweep_id}-{i:04d}"
        click.echo(f"[{i + 1}/{len(configs)}] {this_job_id}")
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        try:
            yaml.dump(cfg, tmp, default_flow_style=False)
            tmp.close()

            results = run_backtest(
                tmp.name, strategy=strategy, backtest_id=this_job_id,
                progress=not no_progress, original_config_path=config,
            )

            job_entry: dict = {
                "job_index": i,
                "job_id": this_job_id,
                "config_params": {k: get_param_value(cfg, k) for k in params},
                "status": "COMPLETED" if results is not None else "NO_RESULTS",
            }

            if results is not None:
                if results.metadata and results.metadata.warnings:
                    for w in results.metadata.warnings:
                        click.echo(f"  warning: {w}", err=True)
                    job_entry["warnings"] = list(results.metadata.warnings)

                if is_s3:
                    job_out_str = f"{out_base}/jobs/{i:04d}"
                    results.save(job_out_str)
                    _generate_report(results, job_out_str)
                else:
                    job_out = out_base / "jobs" / f"{i:04d}"
                    results.save(job_out)
                    shutil.copyfile(tmp.name, job_out / "config.yaml")
                    manifest = {
                        "config": "config.yaml",
                        "job_index": i,
                        "sweep_id": sweep_id,
                        "config_params": job_entry["config_params"],
                    }
                    (job_out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                    _generate_report(results, str(job_out))
                    job_entry["output"] = str(job_out)

            jobs_summary.append(job_entry)

        except Exception as e:
            click.echo(f"  error: {e}", err=True)
            jobs_summary.append({
                "job_index": i,
                "job_id": this_job_id,
                "config_params": {k: get_param_value(cfg, k) for k in params},
                "status": "FAILED",
                "error": str(e),
            })
        finally:
            os.unlink(tmp.name)

    completed = sum(1 for j in jobs_summary if j["status"] == "COMPLETED")

    if not is_s3:
        summary = {
            "sweep_id": sweep_id,
            "job_count": len(configs),
            "completed_count": completed,
            "sweep_params": {k: [str(v) for v in vals] for k, vals in params.items()},
            "jobs": jobs_summary,
        }
        out_base.mkdir(parents=True, exist_ok=True)
        (out_base / "sweep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    click.echo(f"sweep complete: {completed}/{len(configs)} completed, results saved to {out_base}")


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
