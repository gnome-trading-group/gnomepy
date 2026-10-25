from __future__ import annotations

import os
from datetime import datetime

import zstandard

from gnomepy.java.enums import RecordType


def default_merged_bucket() -> str:
    stage = os.getenv("STAGE", "prod").lower()
    return f"gnome-market-data-merged-{stage}"


_RECORD_FAMILY = "book"


def build_s3_key(stream_id: int, record_type: RecordType, dt: datetime) -> str:
    """Build the S3 key StreamEntry.getKey() gives an AGGREGATED minute (zero-padded, so keys sort by time)."""
    return (
        f"{stream_id}/{_RECORD_FAMILY}/{record_type.value}"
        f"/{dt.year:04d}/{dt.month:02d}/{dt.day:02d}/{dt.hour:02d}/{dt.minute:02d}.zst"
    )


def compress(data: bytes) -> bytes:
    return zstandard.ZstdCompressor().compress(data)


def upload(s3_client, bucket: str, key: str, compressed_data: bytes) -> None:
    s3_client.put_object(Bucket=bucket, Key=key, Body=compressed_data)
