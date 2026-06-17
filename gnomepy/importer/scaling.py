from __future__ import annotations

import pandas as pd

_PRICE_SCALE = 1_000_000_000
_SIZE_SCALE = 1_000_000


def scale_price(value: float | int) -> int:
    return int(round(float(value) * _PRICE_SCALE))


def scale_size(value: float | int) -> int:
    return int(round(float(value) * _SIZE_SCALE))


def parse_timestamp_ns(value, fmt: str, tz: str | None = None) -> int:
    """Convert a value to nanoseconds since UTC epoch.

    fmt: "epoch_s", "epoch_ms", "epoch_us", "epoch_ns", "iso8601", or a strftime pattern.
    tz:  timezone name for tz-naive string sources (e.g. "US/Eastern"). Ignored for epoch formats.
    """
    if fmt == "epoch_ns":
        return int(value)
    if fmt == "epoch_us":
        return int(float(value) * 1_000)
    if fmt == "epoch_ms":
        return int(float(value) * 1_000_000)
    if fmt == "epoch_s":
        return int(float(value) * 1_000_000_000)

    # String / datetime-like formats
    ts = pd.to_datetime(value, format=None if fmt == "iso8601" else fmt, utc=False)
    if ts.tzinfo is None and tz is not None:
        ts = ts.tz_localize(tz).tz_convert("UTC")
    elif ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    # pd.Timestamp.value is nanoseconds since epoch
    return ts.value


def apply_price_transform(series: pd.Series) -> pd.Series:
    return (series.astype(float) * _PRICE_SCALE).round().astype("int64")


def apply_size_transform(series: pd.Series) -> pd.Series:
    return (series.astype(float) * _SIZE_SCALE).round().astype("int64")


def apply_timestamp_transform(series: pd.Series, fmt: str, tz: str | None) -> pd.Series:
    """Vectorized timestamp conversion → int64 nanoseconds since epoch."""
    if fmt == "epoch_ns":
        return series.astype("int64")
    if fmt == "epoch_us":
        return (series.astype(float) * 1_000).round().astype("int64")
    if fmt == "epoch_ms":
        return (series.astype(float) * 1_000_000).round().astype("int64")
    if fmt == "epoch_s":
        return (series.astype(float) * 1_000_000_000).round().astype("int64")

    # String-based: use pd.to_datetime then extract ns value
    fmt_arg = None if fmt == "iso8601" else fmt
    parsed = pd.to_datetime(series, format=fmt_arg, utc=False)
    if parsed.dt.tz is None and tz is not None:
        parsed = parsed.dt.tz_localize(tz).dt.tz_convert("UTC")
    elif parsed.dt.tz is not None:
        parsed = parsed.dt.tz_convert("UTC")
    return parsed.astype("int64")
