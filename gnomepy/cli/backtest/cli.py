"""CLI entry point for running backtests.

Usage::

    gnome-backtest config.yaml
    gnome-backtest s3://bucket/config.yaml --dry-run
    gnome-backtest config.yaml --start 2025-01-01T00:00:00 --end 2025-01-02T00:00:00
"""

import argparse
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime

import yaml

from gnomepy.cli.backtest.config import BacktestConfig, load_config
from gnomepy.cli.backtest.factory import build_backtest

logger = logging.getLogger("gnomepy.backtest.cli")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="gnome-backtest",
        description="Run a backtest from a YAML configuration file.",
    )
    parser.add_argument("config", help="Path to YAML config (local or s3:// URI)")
    parser.add_argument("--start", dest="start", default=None, help="Override start_datetime (ISO format)")
    parser.add_argument("--end", dest="end", default=None, help="Override end_datetime (ISO format)")
    parser.add_argument("--listings", dest="listings", default=None, help="Override listing_ids (comma-separated)")
    parser.add_argument("--dry-run", action="store_true", help="Validate and construct without running")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    logger.info("Loading config from %s", args.config)
    config = load_config(args.config)

    # Apply CLI overrides
    if args.start:
        config.backtest.start_datetime = args.start
    if args.end:
        config.backtest.end_datetime = args.end
    if args.listings:
        config.backtest.listing_ids = [int(x.strip()) for x in args.listings.split(",")]

    logger.info("Building backtest object graph")
    backtest = build_backtest(config)

    if args.dry_run:
        logger.info("Dry run complete — config is valid and all objects constructed successfully")
        return

    logger.info("Preparing data")
    backtest.prepare_data()

    logger.info("Executing backtest")
    backtest.fully_execute()

    logger.info("Backtest complete")
    _upload_results(backtest, config)


def _upload_results(backtest, config: BacktestConfig) -> None:
    bucket = config.output.s3_bucket
    prefix = config.output.s3_prefix

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save results NPZ
        npz_path = os.path.join(tmpdir, "results.npz")
        backtest.recorder.to_npz(npz_path)

        # Compute and save performance summary
        summary = {}
        for listing_id in backtest.recorder.market_recorder.listing_id_to_asset_no:
            record = backtest.recorder.get_record(listing_id)
            stats = record.stats()
            summary[str(listing_id)] = stats.get_performance_summary()

        summary_path = os.path.join(tmpdir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Save original config
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config.model_dump(), f, default_flow_style=False)

        if bucket:
            _upload_to_s3(bucket, prefix, run_id, tmpdir)
        else:
            # No S3 configured — save locally
            local_dir = os.path.join("backtest_results", run_id)
            os.makedirs(local_dir, exist_ok=True)
            for fname in ("results.npz", "summary.json", "config.yaml"):
                src = os.path.join(tmpdir, fname)
                dst = os.path.join(local_dir, fname)
                with open(src, "rb") as sf, open(dst, "wb") as df:
                    df.write(sf.read())
            logger.info("Results saved locally to %s", local_dir)

    logger.info("run_id: %s", run_id)


def _upload_to_s3(bucket: str, prefix: str, run_id: str, tmpdir: str) -> None:
    import boto3

    s3 = boto3.client("s3")
    s3_prefix = f"{prefix.rstrip('/')}/{run_id}"

    for fname in ("results.npz", "summary.json", "config.yaml"):
        local = os.path.join(tmpdir, fname)
        key = f"{s3_prefix}/{fname}"
        s3.upload_file(local, bucket, key)
        logger.info("Uploaded s3://%s/%s", bucket, key)

    logger.info("Results uploaded to s3://%s/%s/", bucket, s3_prefix)


if __name__ == "__main__":
    main()
