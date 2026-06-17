from __future__ import annotations

import pandas as pd

from gnomepy.importer.mapping import ImportConfig


def validate(config: ImportConfig, df: pd.DataFrame) -> list[str]:
    """Return a list of error strings. Empty list means the config is valid for this DataFrame."""
    errors: list[str] = []

    # Check all source columns exist in the DataFrame
    for m in config.field_mappings:
        if m.source_column not in df.columns:
            errors.append(f"source_column {m.source_column!r} not found in data (columns: {list(df.columns)})")

    # Check timestamp field is mapped
    ts_mappings = [m for m in config.field_mappings if m.target_field == config.timestamp_field]
    if not ts_mappings:
        errors.append(
            f"timestamp_field {config.timestamp_field!r} has no FieldMapping — "
            "add a FieldMapping with target_field matching timestamp_field"
        )

    # Check timestamp mappings have required metadata
    for m in config.field_mappings:
        if m.transform == "timestamp" and m.timestamp_format is None:
            errors.append(f"FieldMapping for {m.target_field!r} has transform='timestamp' but no timestamp_format")
        if m.transform == "enum" and not m.enum_map:
            errors.append(f"FieldMapping for {m.target_field!r} has transform='enum' but no enum_map")

    if errors:
        return errors

    # Check timestamp column has no nulls
    ts_col = ts_mappings[0].source_column
    if ts_col in df.columns:
        null_count = df[ts_col].isna().sum()
        if null_count > 0:
            errors.append(f"timestamp column {ts_col!r} has {null_count} null values")

    return errors
