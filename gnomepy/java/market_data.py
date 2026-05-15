from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

import jpype

from gnomepy.java._jvm import ensure_jvm_started
from gnomepy.java.cache import MarketDataCache
from gnomepy.java.datastore import DataStore
from gnomepy.java.enums import SchemaType

logger = logging.getLogger(__name__)

_MarketDataEntry = None
_EntryType = None
_LocalDateTime = None
_S3Client = None
_GetObjectRequest = None

_UNSET = object()


def _resolve_classes():
    global _MarketDataEntry, _EntryType, _LocalDateTime, _S3Client, _GetObjectRequest

    if _MarketDataEntry is not None:
        return

    _MarketDataEntry = jpype.JClass("group.gnometrading.data.MarketDataEntry")
    _EntryType = jpype.JClass("group.gnometrading.data.MarketDataEntry$EntryType")
    _LocalDateTime = jpype.JClass("java.time.LocalDateTime")
    _S3Client = jpype.JClass("software.amazon.awssdk.services.s3.S3Client")
    _GetObjectRequest = jpype.JClass("software.amazon.awssdk.services.s3.model.GetObjectRequest")


def _to_java_datetime(dt: datetime):
    _resolve_classes()
    return _LocalDateTime.of(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)


def _create_default_s3_client():
    _resolve_classes()
    return _S3Client.create()


class MarketDataClient:
    """Load market data from S3 using the Java MarketDataEntry infrastructure.

    Requires the JVM to be started with the appropriate JARs on the classpath.
    AWS credentials are resolved via the standard Java SDK credential chain
    (env vars, ~/.aws/credentials, IAM role, etc.).

    Downloaded bytes are cached to local disk by default at
    ``~/.gnomepy/cache/market_data/`` (or ``$GNOMEPY_CACHE_DIR``).
    Pass ``cache_dir=None`` to disable caching.

    Usage:
        from gnomepy.java import ensure_jvm_started, MarketDataClient, SchemaType

        ensure_jvm_started()
        client = MarketDataClient()
        schemas = client.load(
            security_id=1,
            exchange_id=2,
            schema_type=SchemaType.MBP_10,
            start=datetime(2024, 1, 15, 9, 30),
            end=datetime(2024, 1, 15, 16, 0),
        )
    """

    def __init__(self, bucket: str | None = None, s3_client=None, cache_dir=_UNSET):
        """
        Args:
            bucket: S3 bucket name containing market data.
            s3_client: A Java S3Client instance. If None, creates a default client.
            cache_dir: Local cache directory. Omit for default, pass None to disable.
        """
        ensure_jvm_started()
        _resolve_classes()
        self._bucket = bucket or f"gnome-market-data-{os.getenv('STAGE', 'prod').lower()}"
        self._s3_client = s3_client or _create_default_s3_client()

        if cache_dir is _UNSET:
            self._cache: MarketDataCache | None = MarketDataCache()
        elif cache_dir is None:
            self._cache = None
        else:
            self._cache = MarketDataCache(cache_dir)

    def get_java_entries(
        self,
        security_id: int,
        exchange_id: int,
        schema_type: SchemaType,
        start: datetime,
        end: datetime,
    ) -> list:
        """Get raw Java MarketDataEntry objects for use with BacktestDriver.

        Returns a list of Java MarketDataEntry objects (not loaded).
        """
        java_schema_type = schema_type.to_java()
        entries = []
        current = start.replace(second=0, microsecond=0)

        while current < end:
            java_dt = _to_java_datetime(current)
            entries.append(_MarketDataEntry(
                int(security_id),
                int(exchange_id),
                java_schema_type,
                java_dt,
                _EntryType.AGGREGATED,
            ))
            current += timedelta(minutes=1)

        return entries

    def load(
        self,
        security_id: int,
        exchange_id: int,
        schema_type: SchemaType,
        start: datetime,
        end: datetime,
    ) -> DataStore:
        """Load market data from S3 (or local cache) for a time range.

        Args:
            security_id: Security identifier.
            exchange_id: Exchange identifier.
            schema_type: Type of market data schema.
            start: Start datetime (inclusive).
            end: End datetime (exclusive).

        Returns:
            DataStore wrapping the loaded schemas.
        """
        entries = self.get_java_entries(security_id, exchange_id, schema_type, start, end)
        all_data = bytearray()
        hits = 0
        misses = 0

        for entry in entries:
            key = str(entry.getKey())

            if self._cache is not None:
                cached = self._cache.get(key)
                if cached is not None:
                    all_data.extend(cached)
                    hits += 1
                    continue

            try:
                raw = self._download_raw(key)
                if self._cache is not None:
                    self._cache.put(key, raw)
                all_data.extend(raw)
                misses += 1
            except Exception as e:
                if "NoSuchKey" in str(e):
                    continue
                raise RuntimeError(
                    f"Failed to load S3 key: {key} from bucket: {self._bucket}"
                ) from e

        if hits or misses:
            logger.debug("market data cache: %d hits, %d misses", hits, misses)

        return DataStore(bytes(all_data), schema_type)

    def _download_raw(self, key: str) -> bytes:
        request = _GetObjectRequest.builder().bucket(self._bucket).key(key).build()
        response = self._s3_client.getObjectAsBytes(request)
        return bytes(response.asByteArray())
