"""CLI entry point for model training.

Usage::

    gnome-train config.yaml
    gnome-train config.yaml --dry-run
    gnome-train config.yaml --skip-tuning
"""

import argparse
import logging
import os

from gnomepy.cli.train.config import load_train_config
from gnomepy.cli.train.factory import build_trainer

logger = logging.getLogger("gnomepy.cli.train")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="gnome-train",
        description="Train models from a YAML configuration file.",
    )
    parser.add_argument("config", help="Path to YAML config (local or s3:// URI)")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without training")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    logger.info("Loading training config from %s", args.config)
    config = load_train_config(args.config)

    logger.info("Building trainer (model type: %s)", config.model.type)
    trainer = build_trainer(config)

    if args.dry_run:
        logger.info("Dry run complete — config is valid and trainer constructed successfully")
        return

    from gnomepy.research.models.registry import ModelRegistry

    registry = ModelRegistry(base_dir=config.registry.base_dir)

    # Tune unless skipped or explicit params provided
    train_params = config.training.params
    if train_params is None and not args.skip_tuning and config.tuning.enabled:
        logger.info("Starting hyperparameter tuning")
        tuning_kwargs = {
            "train_window": config.tuning.train_window,
            "val_window": config.tuning.val_window,
            "metric": config.tuning.metric,
            "num_boost_round": config.tuning.num_boost_round,
        }
        if config.tuning.n_random is not None:
            tuning_kwargs["n_random"] = config.tuning.n_random
        if config.tuning.param_grid is not None:
            tuning_kwargs["param_grid"] = config.tuning.param_grid

        tuning_result = trainer.tune_hyperparameters(**tuning_kwargs)
        logger.info("Best params: %s", tuning_result.best_params)

    # Train
    logger.info("Training final model")
    model = trainer.train(
        params=train_params,
        num_boost_round=config.training.num_boost_round,
        early_stopping_rounds=config.training.early_stopping_rounds,
        register=True,
        registry=registry,
    )

    # Upload to S3 if configured
    if config.registry.upload_s3:
        _upload_registry_to_s3(config.registry.base_dir, config.registry.s3_bucket, config.registry.s3_prefix)

    logger.info("Training complete")


def _upload_registry_to_s3(base_dir: str, s3_bucket: str | None, s3_prefix: str) -> None:
    if not s3_bucket:
        logger.warning("upload_s3 is true but no s3_bucket configured, skipping upload")
        return

    import boto3

    s3 = boto3.client("s3")
    for root, _dirs, files in os.walk(base_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, base_dir)
            key = f"{s3_prefix.rstrip('/')}/{rel_path}"
            s3.upload_file(local_path, s3_bucket, key)
            logger.info("Uploaded s3://%s/%s", s3_bucket, key)

    logger.info("Registry uploaded to s3://%s/%s", s3_bucket, s3_prefix)


if __name__ == "__main__":
    main()
