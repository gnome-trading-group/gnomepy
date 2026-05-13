"""Event log table builder for the backtest explorer."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from gnomepy.explorer.data import WindowedData

def build_event_columns(price_decimals: int = 2) -> list[dict]:
    price_spec = f".{price_decimals}f"
    return [
        {"name": "Time", "id": "time", "type": "text"},
        {"name": "Type", "id": "type", "type": "text"},
        {"name": "OType", "id": "order_type", "type": "text"},
        {"name": "OID", "id": "oid", "type": "text"},
        {"name": "Src", "id": "source", "type": "text"},
        {"name": "Side", "id": "side", "type": "text"},
        {"name": "Price", "id": "price", "type": "numeric", "format": {"specifier": price_spec}},
        {"name": "Qty", "id": "qty", "type": "numeric", "format": {"specifier": price_spec}},
        {"name": "Fee", "id": "fee", "type": "numeric", "format": {"specifier": price_spec}},
        {"name": "Slip bps", "id": "slippage_bps", "type": "numeric", "format": {"specifier": ".2f"}},
        {"name": "Slip $", "id": "slippage_usd", "type": "numeric", "format": {"specifier": price_spec}},
        {"name": "Status", "id": "status", "type": "text"},
    ]


def build_event_records(
    windowed_a: WindowedData,
    windowed_b: WindowedData | None = None,
) -> list[dict]:
    rows: list[dict] = []
    rows.extend(_fill_records(windowed_a.fills, source="A"))
    rows.extend(_intent_records(windowed_a.intents, source="A"))
    rows.extend(_order_records(windowed_a.orders, source="A"))
    if windowed_b is not None:
        rows.extend(_fill_records(windowed_b.fills, source="B"))
        rows.extend(_intent_records(windowed_b.intents, source="B"))
        rows.extend(_order_records(windowed_b.orders, source="B"))
    rows.sort(key=lambda r: r.get("_ts", 0))
    for r in rows:
        r.pop("_ts", None)
    return rows


def _ts_str(ts: pd.Timestamp) -> str:
    return ts.strftime("%H:%M:%S.%f")


def _fill_records(fills: pd.DataFrame, source: str) -> list[dict]:
    if fills.empty:
        return []
    rows = []
    for ts, row in fills.iterrows():
        rows.append({
            "_ts": ts.value,
            "timestamp_iso": ts.isoformat(),
            "time": _ts_str(ts),
            "type": "fill",
            "order_type": None,
            "oid": str(int(row["client_oid"])) if "client_oid" in row else None,
            "source": source,
            "side": str(row.get("side", "")),
            "price": float(row.get("fill_price", 0)),
            "qty": float(row.get("fill_qty", 0)),
            "fee": float(row.get("fee", 0)),
            "slippage_bps": float(row["slippage_bps"]) if "slippage_bps" in row else None,
            "slippage_usd": float(row["slippage_usd"]) if "slippage_usd" in row else None,
            "status": None,
        })
    return rows


def _intent_records(intents: pd.DataFrame, source: str) -> list[dict]:
    if intents.empty:
        return []
    rows = []
    for ts, row in intents.iterrows():
        bid_p = row.get("bid_price", 0)
        ask_p = row.get("ask_price", 0)
        take_p = row.get("take_limit_price", 0)
        if bid_p > 0 or ask_p > 0:
            price = (bid_p + ask_p) / 2.0 if bid_p > 0 and ask_p > 0 else max(bid_p, ask_p)
            qty = (row.get("bid_size", 0) + row.get("ask_size", 0))
            side = "Bid/Ask" if bid_p > 0 and ask_p > 0 else ("Bid" if bid_p > 0 else "Ask")
        elif take_p > 0:
            price = float(take_p)
            qty = float(row.get("take_size", 0))
            side = "Take"
        else:
            continue
        rows.append({
            "_ts": ts.value,
            "timestamp_iso": ts.isoformat(),
            "time": _ts_str(ts),
            "type": "intent",
            "order_type": None,
            "oid": None,
            "source": source,
            "side": side,
            "price": float(price),
            "qty": float(qty),
            "fee": None,
            "slippage_bps": None,
            "slippage_usd": None,
            "status": None,
        })
    return rows


def _order_records(orders: pd.DataFrame, source: str) -> list[dict]:
    if orders.empty:
        return []
    rows = []
    for ts, row in orders.iterrows():
        avg_fill = float(row.get("avg_fill_price", 0))
        rows.append({
            "_ts": ts.value,
            "timestamp_iso": ts.isoformat(),
            "time": _ts_str(ts),
            "type": "order",
            "order_type": str(row.get("order_type", "")) or None,
            "oid": str(int(row["client_oid"])) if "client_oid" in row else None,
            "source": source,
            "side": str(row.get("side", "")),
            "price": avg_fill if avg_fill > 0 else float(row.get("submit_price", 0)),
            "qty": float(row.get("filled_qty", 0)) or float(row.get("submit_size", 0)),
            "fee": float(row.get("total_fee", 0)),
            "slippage_bps": None,
            "slippage_usd": None,
            "status": str(row.get("final_status", "")),
        })
    return rows
