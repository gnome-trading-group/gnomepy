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

import boto3

from gnomepy.auth import get_id_token as _get_id_token
from gnomepy.auth import login as _auth_login
from gnomepy.auth import logout as _auth_logout
from gnomepy.config import _STAGE
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
@click.option("--compare", default=None, metavar="PATH_OR_RUN_ID", help="Second backtest path or run_id for side-by-side comparison")
@click.option("--compare-job", "compare_job_index", default=0, show_default=True, help="Job index for the comparison run (used when --compare is a run_id)")
@click.option("--run-id", default=None, metavar="RUN_ID", help="Explore a remote run by run_id (looks up S3 path automatically)")
@click.option("--job", "job_index", default=0, show_default=True, help="Job index within a remote run (used with --run-id)")
@click.option("--port", default=8050, show_default=True, help="Port for the Dash server")
@click.option("--no-browser", is_flag=True, help="Do not auto-open the browser")
@click.option("--debug", is_flag=True, hidden=True)
@click.option("--price-decimals", "price_decimals", default=2, show_default=True, help="Decimal places for price display (e.g. 8 for BTC)")
def explore(
    path: str | None,
    compare: str | None,
    compare_job_index: int,
    run_id: str | None,
    job_index: int,
    port: int,
    no_browser: bool,
    debug: bool,
    price_decimals: int,
) -> None:
    """Launch the interactive backtest explorer.

    Accepts a local directory, an S3 URI, or a remote run_id:

    \b
      gnomepy explore --run-id <run_id_a> --compare <run_id_b>
      gnomepy explore --run-id <run_id> [--job 2]
      gnomepy explore ./019e2174-...
      gnomepy explore s3://gnome-research-prod/backtests/<run_id>/jobs/0
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

    launch_explorer(results_a, results_b=results_b, port=port, open_browser=not no_browser, debug=debug, price_decimals=price_decimals)


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
# Cache commands
# ---------------------------------------------------------------------------

@main.group()
def cache() -> None:
    """Manage the local market data cache."""


@cache.command(name="info")
def cache_info() -> None:
    """Show cache location, file count, and total size."""
    from gnomepy.java.cache import MarketDataCache
    c = MarketDataCache()
    total_bytes, file_count = c.size()
    click.echo(f"location: {c._root}")
    click.echo(f"files:    {file_count}")
    click.echo(f"size:     {_human_size(total_bytes)}")


@cache.command(name="clear")
@click.option("--prefix", default=None, help="Only clear keys matching this prefix (e.g. 'mbo/1/2')")
@click.confirmation_option(prompt="Delete cached market data?")
def cache_clear(prefix: str | None) -> None:
    """Delete cached market data files."""
    from gnomepy.java.cache import MarketDataCache
    c = MarketDataCache()
    count = c.clear(prefix=prefix)
    click.echo(f"Deleted {count} cached file(s).")


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


# ---------------------------------------------------------------------------
# Import commands
# ---------------------------------------------------------------------------

@main.group("import")
def import_cmd() -> None:
    """Import historical market data from external vendors."""


@import_cmd.command("tardis")
@click.option("--exchange", required=True, help="Tardis exchange name (e.g., binance-futures, deribit)")
@click.option("--symbols", required=True, help="Comma-separated symbols (e.g., BTCUSDT,ETHUSDT)")
@click.option("--start", required=True, type=click.DateTime(formats=["%Y-%m-%d"]), help="Start date inclusive")
@click.option("--end", required=True, type=click.DateTime(formats=["%Y-%m-%d"]), help="End date inclusive")
@click.option("--dry-run", is_flag=True, help="Validate without uploading to S3")
@click.option("--bucket", default=None, help="Override S3 bucket")
def import_tardis(
    exchange: str,
    symbols: str,
    start,
    end,
    dry_run: bool,
    bucket: str | None,
) -> None:
    """Import Tardis incremental L2 + trades data as MBP_10 into gnome market data."""
    from gnomepy.importer.tardis import TardisImporter, TardisImportRequest

    request = TardisImportRequest(
        exchange=exchange,
        symbols=[s.strip() for s in symbols.split(",")],
        start_date=start.date(),
        end_date=end.date(),
        bucket=bucket,
        dry_run=dry_run,
    )
    results = TardisImporter().run(request)
    for r in results:
        click.echo(f"{r.exchange} / {r.symbol}  (security_id={r.security_id}, exchange_id={r.exchange_id})")
        click.echo(f"  processed: {r.days_processed}  skipped: {r.days_skipped}  records: {r.total_records}")
        for err in r.errors:
            click.echo(f"  error: {err}", err=True)


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
@click.option("--no-cache", is_flag=True, help="Disable local market data caching")
def backtest_run(
    config: str,
    strategy_path: str | None,
    java_strategy: str | None,
    output: str | None,
    job_id: str | None,
    no_progress: bool,
    no_cache: bool,
) -> None:
    """Run a backtest locally and generate a report."""
    if strategy_path and java_strategy:
        raise click.UsageError("pass either --strategy or --java-strategy, not both")

    strategy = strategy_path or java_strategy or None
    parsed = yaml.safe_load(Path(config).read_text())
    params = sweep_params(parsed)
    use_cache = not no_cache

    if not params:
        _run_single(config, strategy, output, job_id or uuid7(), no_progress, use_cache)
    else:
        _run_sweep(config, parsed, params, strategy, output, job_id, no_progress, use_cache)


def _run_single(config: str, strategy, output: str | None, job_id: str, no_progress: bool, cache: bool = True) -> None:
    click.echo(f"job_id: {job_id}")
    results = run_backtest(config, strategy=strategy, backtest_id=job_id, progress=not no_progress, cache=cache)
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
    cache: bool = True,
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
                cache=cache,
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


def _download_job_files(bucket: str, s3_prefix: str, local_dir: Path) -> int:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(s3_prefix):].lstrip("/")
            local_path = local_dir / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(local_path))
            count += 1
    return count


@backtest.command(name="results")
@click.argument("run_id")
@click.option("--output", "-o", default=None, type=click.Path(), help="Download all result files to this directory")
def backtest_results(run_id: str, output: str | None) -> None:
    """Show report URLs or download all result files for a completed backtest run."""
    run = get_backtest(run_id)
    jobs = run.get("jobs", [])

    if output is None:
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
        return

    bucket = f"gnome-research-{_STAGE}"
    out_root = Path(output)
    downloaded = 0
    for job in jobs:
        status = job.get("status", "")
        idx = job.get("array_index", 0)
        if status not in ("COMPLETED", "SUCCEEDED"):
            click.echo(f"[{idx:04d}] skipping — status={status}")
            continue
        s3_prefix = f"backtests/{run_id}/jobs/{idx}/"
        local_dir = out_root / "jobs" / f"{idx:04d}"
        count = _download_job_files(bucket, s3_prefix, local_dir)
        click.echo(f"[{idx:04d}] {count} files → {local_dir}")
        downloaded += count
    click.echo(f"done — {downloaded} total files downloaded to {out_root}")


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
