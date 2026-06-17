from __future__ import annotations

from typing import Any

import pandas as pd

from gnomepy.importer.mapping import FieldMapping, ImportConfig
from gnomepy.importer.scaling import parse_timestamp_ns, scale_price, scale_size


def _apply_mapping(value, mapping: FieldMapping) -> Any:
    if pd.isna(value):
        return None
    if mapping.transform == "none":
        return int(value)
    if mapping.transform == "price":
        return scale_price(value)
    if mapping.transform in ("size", "volume"):
        return scale_size(value)
    if mapping.transform == "timestamp":
        return parse_timestamp_ns(value, mapping.timestamp_format, mapping.timestamp_tz)
    if mapping.transform == "enum":
        mapped = mapping.enum_map.get(str(value))
        if mapped is None:
            raise ValueError(f"enum_map has no entry for value {value!r} in field {mapping.target_field!r}")
        return mapped
    raise ValueError(f"Unknown transform {mapping.transform!r} for field {mapping.target_field!r}")


def encode_chunk(chunk: pd.DataFrame, config: ImportConfig) -> bytes:
    """Encode a minute-chunk DataFrame to concatenated SBE bytes.

    Requires the JVM to be started before calling.
    """
    from gnomepy.java.schemas import get_schema_class

    schema_cls = get_schema_class(config.schema_type)
    parts: list[bytes] = []

    for _, row in chunk.iterrows():
        # Build kwargs: config-level IDs, then defaults, then per-row field mappings
        kwargs: dict[str, Any] = {
            "exchange_id": config.exchange_id,
            "security_id": config.security_id,
        }
        kwargs.update(config.defaults)
        for mapping in config.field_mappings:
            kwargs[mapping.target_field] = _apply_mapping(row[mapping.source_column], mapping)

        schema = schema_cls(**kwargs)
        parts.append(schema.encode())

    return b"".join(parts)
