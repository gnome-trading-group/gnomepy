from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from gnomepy.importer.chunker import chunk_by_minute
from gnomepy.importer.encoder import encode_chunk
from gnomepy.importer.mapping import ImportConfig
from gnomepy.importer.scaling import apply_timestamp_transform
from gnomepy.importer.uploader import build_s3_key, compress, default_merged_bucket, upload
from gnomepy.importer.validators import validate


@dataclass
class ImportResult:
    files_uploaded: int
    total_records: int
    minutes_covered: int


@dataclass
class DryRunResult:
    is_valid: bool
    errors: list[str]
    minutes_count: int
    total_records: int
    sample_keys: list[str] = field(default_factory=list)


class ImportJob:
    """Orchestrates converting a CSV/Parquet file into the gnome market data format and uploading to S3."""

    def __init__(self, config: ImportConfig, s3_client=None):
        self.config = config
        self._s3 = s3_client

    def _s3_client(self):
        if self._s3 is None:
            import boto3
            self._s3 = boto3.client("s3")
        return self._s3

    def _load(self, source, file_format: str = "auto") -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            return source.copy()
        path = str(source)
        fmt = file_format
        if fmt == "auto":
            fmt = "parquet" if path.endswith(".parquet") else "csv"
        if fmt == "parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def _timestamp_ns(self, df: pd.DataFrame) -> pd.Series:
        ts_mapping = next(
            m for m in self.config.field_mappings if m.target_field == self.config.timestamp_field
        )
        return apply_timestamp_transform(
            df[ts_mapping.source_column], ts_mapping.timestamp_format, ts_mapping.timestamp_tz
        )

    def _bucket(self) -> str:
        return self.config.bucket or default_merged_bucket()

    def dry_run(self, source, file_format: str = "auto") -> DryRunResult:
        """Validate the config and source data without uploading anything."""
        df = self._load(source, file_format)
        errors = validate(self.config, df)
        if errors:
            return DryRunResult(is_valid=False, errors=errors, minutes_count=0, total_records=0)

        ts_ns = self._timestamp_ns(df)
        chunks = chunk_by_minute(df, ts_ns)
        config = self.config
        sample_keys = [
            build_s3_key(config.security_id, config.exchange_id, config.schema_type, dt)
            for dt in sorted(chunks)[:5]
        ]
        return DryRunResult(
            is_valid=True,
            errors=[],
            minutes_count=len(chunks),
            total_records=len(df),
            sample_keys=sample_keys,
        )

    def run(self, source, file_format: str = "auto") -> ImportResult:
        """Encode source data and upload to S3.

        Validates first; raises ValueError if config is invalid.
        Requires the JVM to be started before calling (for SBE encoding).
        """
        from gnomepy.java._jvm import ensure_jvm_started

        df = self._load(source, file_format)
        errors = validate(self.config, df)
        if errors:
            raise ValueError("Import config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        ensure_jvm_started()
        ts_ns = self._timestamp_ns(df)
        chunks = chunk_by_minute(df, ts_ns)

        bucket = self._bucket()
        s3 = self._s3_client()
        files_uploaded = 0
        total_records = 0

        for minute_dt, chunk_df in sorted(chunks.items()):
            raw = encode_chunk(chunk_df, self.config)
            compressed = compress(raw)
            key = build_s3_key(
                self.config.security_id,
                self.config.exchange_id,
                self.config.schema_type,
                minute_dt,
            )
            upload(s3, bucket, key, compressed)
            files_uploaded += 1
            total_records += len(chunk_df)

        return ImportResult(
            files_uploaded=files_uploaded,
            total_records=total_records,
            minutes_covered=len(chunks),
        )
