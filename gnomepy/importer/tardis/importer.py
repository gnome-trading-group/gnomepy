from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from gnomepy.importer.import_job import ImportJob
from gnomepy.importer.tardis.client import TardisClient
from gnomepy.importer.tardis.mappings import build_mbp10_df, mbp10_import_config

TARDIS_EXCHANGE_NAMES: dict[str, str] = {
    "binance": "Binance",
    "binance-futures": "Binance",
    "deribit": "Deribit",
    "hyperliquid": "Hyperliquid",
    "lighter": "Lighter",
}


@dataclass
class TardisImportRequest:
    exchange: str
    symbols: list[str]
    start_date: date
    end_date: date
    api_key: str | None = None
    bucket: str | None = None
    dry_run: bool = False


@dataclass
class TardisImportResult:
    exchange: str
    symbol: str
    security_id: int
    exchange_id: int
    days_processed: int
    days_skipped: int
    total_records: int
    errors: list[str] = field(default_factory=list)


class TardisImporter:
    """Imports Tardis incremental L2 + trade data as MBP_10 records into gnome market data S3."""

    def __init__(
        self,
        client: TardisClient | None = None,
        registry=None,
        s3_client=None,
    ) -> None:
        self._client = client
        self._registry = registry
        self._s3 = s3_client

    def run(self, request: TardisImportRequest) -> list[TardisImportResult]:
        results = []
        for symbol in request.symbols:
            results.append(self._run_symbol(request, symbol))
        return results

    def _run_symbol(self, request: TardisImportRequest, symbol: str) -> TardisImportResult:
        security_id, exchange_id = self._resolve(request.exchange, symbol)
        config = mbp10_import_config(security_id, exchange_id, bucket=request.bucket)
        job = ImportJob(config, s3_client=self._s3)
        client = self._client or TardisClient(api_key=request.api_key)

        result = TardisImportResult(
            exchange=request.exchange,
            symbol=symbol,
            security_id=security_id,
            exchange_id=exchange_id,
            days_processed=0,
            days_skipped=0,
            total_records=0,
        )

        current = request.start_date
        while current <= request.end_date:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                try:
                    client.download(
                        exchange=request.exchange,
                        data_types=["incremental_book_L2", "trades"],
                        day=current,
                        symbols=[symbol],
                        dest_dir=tmp_path,
                    )
                    l2_df, trades_df = _load_day(tmp_path, request.exchange, symbol, current)
                    if l2_df is None or trades_df is None:
                        result.days_skipped += 1
                        current += timedelta(days=1)
                        continue

                    mbp10_df = build_mbp10_df(l2_df, trades_df)
                    if mbp10_df.empty:
                        result.days_skipped += 1
                        current += timedelta(days=1)
                        continue

                    if request.dry_run:
                        dry = job.dry_run(mbp10_df)
                        if not dry.is_valid:
                            result.errors.extend(dry.errors)
                        result.total_records += dry.total_records
                    else:
                        imported = job.run(mbp10_df)
                        result.total_records += imported.total_records

                    result.days_processed += 1
                except Exception as exc:
                    result.errors.append(f"{current}: {exc}")
                    result.days_skipped += 1

            current += timedelta(days=1)

        return result

    def _resolve(self, tardis_exchange: str, tardis_symbol: str) -> tuple[int, int]:
        gnome_name = TARDIS_EXCHANGE_NAMES.get(tardis_exchange)
        if gnome_name is None:
            raise ValueError(
                f"Tardis exchange {tardis_exchange!r} is not mapped to a gnome exchange. "
                f"Known exchanges: {list(TARDIS_EXCHANGE_NAMES)}"
            )

        from gnomepy.registry.api import RegistryClient
        registry = self._registry or RegistryClient()

        exchanges = registry.get_exchange(exchange_name=gnome_name)
        if not exchanges:
            raise ValueError(f"Exchange {gnome_name!r} not found in gnome registry")
        exchange_id = exchanges[0].exchange_id

        listings = registry.get_listing(
            exchange_id=exchange_id,
            exchange_security_symbol=tardis_symbol,
        )
        if not listings:
            raise ValueError(
                f"Symbol {tardis_symbol!r} not found in gnome registry for exchange {gnome_name!r}. "
                "Register the listing first."
            )
        return listings[0].security_id, exchange_id


def _load_day(
    tmp_path: Path,
    exchange: str,
    symbol: str,
    day: date,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    date_str = day.strftime("%Y-%m-%d")
    l2_files = list(tmp_path.glob(f"*incremental_book_L2*{date_str}*{symbol}*.csv.gz"))
    trade_files = list(tmp_path.glob(f"*trades*{date_str}*{symbol}*.csv.gz"))

    if not l2_files or not trade_files:
        return None, None

    l2_df = pd.read_csv(l2_files[0])
    trades_df = pd.read_csv(trade_files[0])
    return l2_df, trades_df
