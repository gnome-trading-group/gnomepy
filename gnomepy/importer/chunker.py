from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def chunk_by_minute(df: pd.DataFrame, ts_ns: pd.Series) -> dict[datetime, pd.DataFrame]:
    """Split df into per-minute buckets using ts_ns (int64 nanoseconds UTC).

    Returns a dict mapping naive UTC datetime (truncated to minute) → subset of df,
    sorted by timestamp within each bucket.
    """
    _MINUTE_NS = 60_000_000_000

    minute_ns = (ts_ns // _MINUTE_NS) * _MINUTE_NS
    df = df.copy()
    df["__ts_ns"] = ts_ns
    df["__minute_ns"] = minute_ns

    result: dict[datetime, pd.DataFrame] = {}
    for bucket_ns, group in df.groupby("__minute_ns", sort=True):
        group = group.sort_values("__ts_ns").drop(columns=["__ts_ns", "__minute_ns"])
        # Naive UTC datetime matching Java LocalDateTime used in MarketDataEntry
        dt = datetime.fromtimestamp(int(bucket_ns) / 1e9, tz=timezone.utc).replace(tzinfo=None)
        result[dt] = group

    return result
