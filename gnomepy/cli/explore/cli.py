"""CLI entry point for data exploration.

Usage::

    gnome-explore config.yaml
    gnome-explore s3://bucket/explore_config.yaml
"""

import argparse
import logging


logger = logging.getLogger("gnomepy.cli.explore")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="gnome-explore",
        description="Explore market data from a YAML configuration file.",
    )
    parser.add_argument("config", help="Path to YAML config (local or s3:// URI)")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    raise NotImplementedError("gnome-explore is not yet implemented")


if __name__ == "__main__":
    main()
