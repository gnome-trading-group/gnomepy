from __future__ import annotations

import os
from datetime import datetime

import zstandard

from gnomepy.java.enums import SchemaType


def default_merged_bucket() -> str:
    stage = os.getenv("STAGE", "prod").lower()
    return f"gnome-market-data-merged-{stage}"


def build_s3_key(security_id: int, exchange_id: int, schema_type: SchemaType, dt: datetime) -> str:
    """Build S3 key matching MarketDataEntry.getKey() AGGREGATED format (no zero-padding)."""
    return (
        f"{security_id}/{exchange_id}"
        f"/{dt.year}/{dt.month}/{dt.day}/{dt.hour}/{dt.minute}"
        f"/{schema_type.value}.zst"
    )


def compress(data: bytes) -> bytes:
    return zstandard.ZstdCompressor().compress(data)


def upload(s3_client, bucket: str, key: str, compressed_data: bytes) -> None:
    s3_client.put_object(Bucket=bucket, Key=key, Body=compressed_data)
