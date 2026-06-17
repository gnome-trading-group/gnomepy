from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gnomepy.java.enums import SchemaType


@dataclass
class FieldMapping:
    """Maps one vendor column to one SBE schema field.

    transform types:
        "none"      — pass through as int (e.g. a pre-scaled integer field)
        "price"     — float → int64 * 1e9
        "size"      — float → int64 * 1e6  (covers size and volume fields)
        "timestamp" — various formats → int64 nanoseconds since epoch
        "enum"      — string → string via enum_map (e.g. "buy" → "Bid")
    """

    source_column: str
    target_field: str
    transform: str = "none"
    # Required when transform="timestamp"
    timestamp_format: str | None = None  # "epoch_s", "epoch_ms", "epoch_us", "epoch_ns", "iso8601", or strftime
    timestamp_tz: str | None = None      # tz name for tz-naive string sources, e.g. "US/Eastern"
    # Required when transform="enum"
    enum_map: dict[str, str] | None = None


@dataclass
class ImportConfig:
    """Complete configuration for one import job (one security, exchange, schema type)."""

    schema_type: SchemaType
    security_id: int
    exchange_id: int
    field_mappings: list[FieldMapping]
    # Which target_field holds the event timestamp used for minute-chunking
    timestamp_field: str = "timestamp_event"
    # Override the default merged bucket (gnome-market-data-merged-{STAGE})
    bucket: str | None = None
    # Constant values for SBE fields not present in the source data
    defaults: dict[str, Any] = field(default_factory=dict)
