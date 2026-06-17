from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path


class TardisClient:
    """Thin wrapper around the tardis-dev Python package for downloading market data CSV files."""

    def __init__(self, api_key: str | None = None):
        try:
            import tardis_dev  # noqa: F401
        except ImportError:
            raise ImportError(
                "tardis-dev is required for Tardis market data imports. "
                "Install it with: poetry install -E tardis"
            )
        self._api_key = api_key or os.environ.get("TARDIS_API_KEY", "")

    def download(
        self,
        exchange: str,
        data_types: list[str],
        day: date,
        symbols: list[str],
        dest_dir: Path,
    ) -> None:
        """Download all data_types for a single day to dest_dir.

        Uses tardis-dev's download_datasets which handles auth, retries, and file naming.
        """
        from tardis_dev import download_datasets

        from_date = day.strftime("%Y-%m-%d")
        to_date = (day + timedelta(days=1)).strftime("%Y-%m-%d")

        download_datasets(
            exchange=exchange,
            data_types=data_types,
            from_date=from_date,
            to_date=to_date,
            symbols=symbols,
            api_key=self._api_key,
            download_dir=str(dest_dir),
        )

    def get_exchange_details(self, exchange: str) -> dict:
        """Return exchange metadata from the Tardis API (available symbols, date ranges)."""
        from tardis_dev import get_exchange_details
        return get_exchange_details(exchange)
