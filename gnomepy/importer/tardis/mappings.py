from __future__ import annotations

import math

import pandas as pd

from gnomepy.importer.mapping import FieldMapping, ImportConfig
from gnomepy.importer.tardis.book import L2Book
from gnomepy.java.enums import SchemaType

_NUM_LEVELS = 10
_DEPTH_NULL = 255  # Mbp10Encoder.depthNullValue()
_SIDE_MAP = {"buy": "Bid", "sell": "Ask", "unknown": "None_"}


def build_mbp10_df(l2_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct an MBP_10 event stream from Tardis incremental L2 and trades data.

    Processes rows chronologically, maintaining an L2Book. Emits one MBP_10 row per
    book-changing event (with the shallowest changed level as depth) and one per trade
    (with current book state and depth=255, matching live gateway convention).

    Pre-snapshot L2 updates (before the first is_snapshot=True batch) are discarded.
    """
    l2 = l2_df.copy()
    l2["src"] = "l2"
    trades = trades_df.copy()
    trades["src"] = "trade"

    merged = (
        pd.concat([l2, trades], ignore_index=True)
        .sort_values("timestamp", kind="mergesort")
        .reset_index(drop=True)
    )

    book = L2Book()
    rows: list[dict] = []
    in_snapshot = False
    initialized = False
    snapshot_ts: int = 0
    snapshot_local_ts: int = 0

    for row in merged.itertuples(index=False):
        src = row.src

        if src == "l2":
            is_snap = _is_snapshot(row.is_snapshot)

            if is_snap:
                if not in_snapshot:
                    if initialized:
                        book.clear()
                        initialized = False
                    in_snapshot = True
                    snapshot_ts = int(row.timestamp)
                    snapshot_local_ts = int(row.local_timestamp)
                book.update(row.side, float(row.price), float(row.amount))
                continue

            if in_snapshot:
                in_snapshot = False
                initialized = True
                rows.append(_book_row(snapshot_ts, snapshot_local_ts, 0, book))

            if not initialized:
                continue

            depth = book.update(row.side, float(row.price), float(row.amount))
            if depth is not None:
                rows.append(_book_row(int(row.timestamp), int(row.local_timestamp), depth, book))

        else:  # trade
            if in_snapshot:
                in_snapshot = False
                initialized = True
                rows.append(_book_row(snapshot_ts, snapshot_local_ts, 0, book))

            if not initialized:
                continue

            rows.append(_trade_row(row, book))

    if in_snapshot:
        rows.append(_book_row(snapshot_ts, snapshot_local_ts, 0, book))

    return pd.DataFrame(rows)


def mbp10_import_config(security_id: int, exchange_id: int, bucket: str | None = None) -> ImportConfig:
    """Return the ImportConfig for encoding an MBP_10 DataFrame produced by build_mbp10_df."""
    field_mappings = [
        FieldMapping("timestamp", "timestamp_event", transform="timestamp", timestamp_format="epoch_us"),
        FieldMapping("local_timestamp", "timestamp_recv", transform="timestamp", timestamp_format="epoch_us"),
        FieldMapping("trade_price", "price", transform="price"),
        FieldMapping("trade_size", "size", transform="size"),
        FieldMapping(
            "action", "action", transform="enum",
            enum_map={"Trade": "Trade", "None_": "None_"},
        ),
        FieldMapping(
            "trade_side", "side", transform="enum",
            enum_map={"Bid": "Bid", "Ask": "Ask", "None_": "None_"},
        ),
        FieldMapping("depth", "depth", transform="none"),
    ]
    for i in range(_NUM_LEVELS):
        field_mappings += [
            FieldMapping(f"bid_price_{i}", f"bid_price_{i}", transform="price"),
            FieldMapping(f"bid_size_{i}", f"bid_size_{i}", transform="size"),
            FieldMapping(f"ask_price_{i}", f"ask_price_{i}", transform="price"),
            FieldMapping(f"ask_size_{i}", f"ask_size_{i}", transform="size"),
        ]

    return ImportConfig(
        schema_type=SchemaType.MBP_10,
        security_id=security_id,
        exchange_id=exchange_id,
        field_mappings=field_mappings,
        timestamp_field="timestamp_event",
        bucket=bucket,
        defaults={"timestamp_sent": 0},
    )


def _is_snapshot(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _level_fields(book: L2Book) -> dict:
    top_bids, top_asks = book.top_levels()
    fields: dict = {}
    for i in range(_NUM_LEVELS):
        fields[f"bid_price_{i}"] = top_bids[i][0] if i < len(top_bids) else math.nan
        fields[f"bid_size_{i}"] = top_bids[i][1] if i < len(top_bids) else math.nan
        fields[f"ask_price_{i}"] = top_asks[i][0] if i < len(top_asks) else math.nan
        fields[f"ask_size_{i}"] = top_asks[i][1] if i < len(top_asks) else math.nan
    return fields


def _book_row(timestamp: int, local_timestamp: int, depth: int, book: L2Book) -> dict:
    return {
        "timestamp": timestamp,
        "local_timestamp": local_timestamp,
        "trade_price": math.nan,
        "trade_size": math.nan,
        "action": "None_",
        "trade_side": "None_",
        "depth": depth,
        **_level_fields(book),
    }


def _trade_row(row, book: L2Book) -> dict:
    return {
        "timestamp": int(row.timestamp),
        "local_timestamp": int(row.local_timestamp),
        "trade_price": float(row.price),
        "trade_size": float(row.amount),
        "action": "Trade",
        "trade_side": _SIDE_MAP.get(str(row.side), "None_"),
        "depth": _DEPTH_NULL,
        **_level_fields(book),
    }
