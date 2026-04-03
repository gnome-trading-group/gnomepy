from __future__ import annotations

from datetime import datetime, timedelta

import jpype
import pandas as pd

from gnomepy.java._jvm import ensure_jvm_started
from gnomepy.java.enums import SchemaType
from gnomepy.java.schemas import Schema, wrap_schema

_MarketDataEntry = None
_EntryType = None
_LocalDateTime = None
_S3Client = None


def _resolve_classes():
    """Lazily resolve Java classes after JVM is started."""
    global _MarketDataEntry, _EntryType, _LocalDateTime, _S3Client

    if _MarketDataEntry is not None:
        return

    _MarketDataEntry = jpype.JClass("group.gnometrading.data.MarketDataEntry")
    _EntryType = jpype.JClass("group.gnometrading.data.MarketDataEntry$EntryType")
    _LocalDateTime = jpype.JClass("java.time.LocalDateTime")
    _S3Client = jpype.JClass("software.amazon.awssdk.services.s3.S3Client")


def _to_java_datetime(dt: datetime):
    """Convert Python datetime to Java LocalDateTime."""
    _resolve_classes()
    return _LocalDateTime.of(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)


def _create_default_s3_client():
    """Create a default Java S3Client using the standard credential chain."""
    _resolve_classes()
    return _S3Client.create()


class MarketDataClient:
    """Load market data from S3 using the Java MarketDataEntry infrastructure.

    Requires the JVM to be started with the appropriate JARs on the classpath.
    AWS credentials are resolved via the standard Java SDK credential chain
    (env vars, ~/.aws/credentials, IAM role, etc.).

    Usage:
        from gnomepy.java import ensure_jvm_started, MarketDataClient, SchemaType

        ensure_jvm_started()
        client = MarketDataClient(bucket="my-market-data-bucket")
        schemas = client.load(
            security_id=1,
            exchange_id=2,
            schema_type=SchemaType.MBP_10,
            start=datetime(2024, 1, 15, 9, 30),
            end=datetime(2024, 1, 15, 16, 0),
        )
    """

    def __init__(self, bucket: str = "gnome-market-data-prod", s3_client=None):
        """
        Args:
            bucket: S3 bucket name containing market data.
            s3_client: A Java S3Client instance. If None, creates a default client.
        """
        ensure_jvm_started()
        _resolve_classes()
        self._bucket = bucket
        self._s3_client = s3_client or _create_default_s3_client()

    def load(
        self,
        security_id: int,
        exchange_id: int,
        schema_type: SchemaType,
        start: datetime,
        end: datetime,
    ) -> list[Schema]:
        """Load market data from S3 for a time range.

        Iterates minute-by-minute over the range, loading aggregated entries.

        Args:
            security_id: Security identifier.
            exchange_id: Exchange identifier.
            schema_type: Type of market data schema.
            start: Start datetime (inclusive).
            end: End datetime (exclusive).

        Returns:
            List of Schema wrappers around the loaded data.
        """
        java_schema_type = schema_type.to_java()
        results = []
        current = start.replace(second=0, microsecond=0)

        while current < end:
            java_dt = _to_java_datetime(current)
            entry = _MarketDataEntry(
                int(security_id),
                int(exchange_id),
                java_schema_type,
                java_dt,
                _EntryType.AGGREGATED,
            )
            try:
                java_schemas = entry.loadFromS3(self._s3_client, self._bucket)
                for js in java_schemas:
                    results.append(wrap_schema(js))
            except Exception as e:
                msg = str(e)
                if "NoSuchKey" in msg:
                    continue  # Missing minute slot — expected
                raise RuntimeError(
                    f"Failed to load S3 key: {entry.getKey()} from bucket: {self._bucket}"
                ) from e
            current += timedelta(minutes=1)

        return results

    def load_dataframe(
        self,
        security_id: int,
        exchange_id: int,
        schema_type: SchemaType,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Load market data and convert to a pandas DataFrame.

        Args:
            Same as load().

        Returns:
            DataFrame with one row per schema record.
        """
        schemas = self.load(security_id, exchange_id, schema_type, start, end)
        if not schemas:
            return pd.DataFrame()
        rows = [s.to_dict() for s in schemas]
        return pd.DataFrame(rows)

    def get_java_entries(
        self,
        security_id: int,
        exchange_id: int,
        schema_type: SchemaType,
        start: datetime,
        end: datetime,
    ) -> list:
        """Get raw Java MarketDataEntry objects for use with BacktestDriver.

        Returns a list of Java MarketDataEntry objects (not wrapped).
        """
        java_schema_type = schema_type.to_java()
        entries = []
        current = start.replace(second=0, microsecond=0)

        while current < end:
            java_dt = _to_java_datetime(current)
            entry = _MarketDataEntry(
                int(security_id),
                int(exchange_id),
                java_schema_type,
                java_dt,
                _EntryType.AGGREGATED,
            )
            entries.append(entry)
            current += timedelta(minutes=1)

        return entries
