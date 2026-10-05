from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy._fs import (
    fs_exists,
    fs_list_parquets,
    fs_mkdir,
    fs_read_json,
    fs_read_parquet,
    fs_write_json,
    fs_write_parquet,
    resolve_fs,
)
from gnomepy.java.statics import Scales
from gnomepy.metadata import BacktestMetadata

# Byte-encoding constants mirrored from BacktestRecorder.java
_SIDE_MAP = {0: "None", 1: "Bid", 2: "Ask"}
_OTYPE_MAP = {0: "Limit", 1: "Market"}
_STATUS_MAP = {0: "Filled", 1: "PartialFill", 2: "Cancelled", 3: "Rejected", 4: "Expired", 5: "Open"}


# SBE null for int64 price and size fields: the recorder writes it for an empty book side or a tick with no trade.
_NULL_LONG = np.iinfo(np.int64).min

_STREAMS = ("market", "orders", "fills", "intents")

# (price columns, size columns) per stream. Market columns depend on the recorded depth, so they are matched by name.
_UNIT_COLUMNS = {
    "orders": (
        ("submit_price", "final_price", "avg_fill_price"),
        ("submit_size", "final_size", "filled_qty", "leaves_qty"),
    ),
    "fills": (("fill_price", "book_bid_price", "book_ask_price"), ("fill_qty", "leaves_qty")),
    "intents": (("bid_price", "ask_price", "take_limit_price"), ("bid_size", "ask_size", "take_size")),
}


def _unit_columns(stream: str, df: pd.DataFrame) -> tuple[list[str], list[str]]:
    if stream == "market":
        prices = [c for c in df.columns if c.startswith(("bid_price_", "ask_price_")) or c == "last_trade_price"]
        sizes = [c for c in df.columns if c.startswith(("bid_size_", "ask_size_")) or c == "last_trade_size"]
        return prices, sizes
    prices, sizes = _UNIT_COLUMNS[stream]
    return [c for c in prices if c in df.columns], [c for c in sizes if c in df.columns]


def _null_to_nan(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        values = df[col]
        if (values == _NULL_LONG).any():
            df[col] = values.astype(float).where(values != _NULL_LONG)


def _rescale(df: pd.DataFrame, stream: str, to_scaled: bool) -> pd.DataFrame:
    """Converts a stream between raw units (1e9 price, 1e6 size) and scaled ones (dollars, units)."""
    if df.empty:
        return df
    prices, sizes = _unit_columns(stream, df)
    out = df.copy()
    for columns, scale in ((prices, Scales.PRICE), (sizes, Scales.SIZE)):
        for col in columns:
            if to_scaled:
                out[col] = out[col] / scale
                continue
            raw = np.rint(out[col] * scale)
            out[col] = raw if raw.isna().any() else raw.astype(np.int64)
    return out


def _decode_bytes(arr: np.ndarray, mapping: dict) -> list[str]:
    return [mapping.get(int(v), "Unknown") for v in arr]


def _buffer_to_df(buffer) -> pd.DataFrame:
    """Convert a Java RecordBuffer to a pandas DataFrame.

    Each ColumnDef in the buffer becomes one column. The column name and type
    are taken from the ColumnDef metadata; no scaling is applied here.
    """
    n = int(buffer.getCount())
    if n == 0:
        return pd.DataFrame()

    data = {}
    for col_def in buffer.getColumns():
        col_name = str(col_def.name())
        col_type = str(col_def.type())
        col_idx = int(col_def.columnIndex())
        if col_type == "LONG":
            data[col_name] = np.array(buffer.getLongColumn(col_idx)[:n], dtype=np.int64)
        elif col_type == "DOUBLE":
            data[col_name] = np.array(buffer.getDoubleColumn(col_idx)[:n], dtype=np.float64)
        elif col_type == "INT":
            data[col_name] = np.array(buffer.getIntColumn(col_idx)[:n], dtype=np.int32)
        elif col_type == "BYTE":
            data[col_name] = np.frombuffer(bytes(buffer.getByteColumn(col_idx)[:n]), dtype=np.uint8)
        elif col_type == "STRING":
            data[col_name] = list(buffer.getStringColumn(col_idx)[:n])

    return pd.DataFrame(data)


class MetricRecorder:
    """Python wrapper around Java MetricRecorder.

    Passed to strategies via ``self.metrics`` inside ``register_metrics()``.
    Strategies create named RecordBuffers, add typed columns, freeze the buffer,
    then write rows during ``on_market_data`` / ``on_execution_report``.

    Usage::

        def register_metrics(self):
            buf = self.metrics.create_buffer("signals")
            self.ts_col  = buf.add_long_column("timestamp")
            self.fv_col  = buf.add_double_column("fair_value")
            buf.freeze()
            self.buf = buf

        def on_market_data(self, data):
            row = self.buf.append_row()
            self.buf.set_long(row, self.ts_col, data.event_timestamp)
            self.buf.set_double(row, self.fv_col, self.compute_fair_value())
            return []
    """

    def __init__(self, java_metric_recorder):
        self._java = java_metric_recorder

    def create_buffer(self, name: str, initial_capacity: int = 1_000_000):
        """Create a new named RecordBuffer for custom metrics.

        After adding columns and calling ``freeze()``, write rows via the
        buffer directly (no Python wrapper — call Java methods on the
        returned object).

        Args:
            name: Unique name for this stream; used as the parquet file name.
            initial_capacity: Initial row capacity; grows automatically.

        Returns:
            Java RecordBuffer instance (call Java methods directly via JPype).
        """
        return self._java.createBuffer(name, initial_capacity)


class BacktestResults:
    """Data access layer for backtest recorder output.

    Converts columnar Java RecordBuffers into cached pandas DataFrames.

    Built-in streams (populated automatically):
      - market_records_df()  — one row per market data tick; BBO + configurable depth
      - orders_df()          — one row per order (written at terminal state)
      - fills_df()           — one row per fill event with BBO context for slippage
      - intent_records_df()  — one row per strategy intent

    Custom strategy streams:
      - custom_metrics(name) — DataFrame for a named custom RecordBuffer
      - custom_metrics()     — dict of all custom stream DataFrames

    Usage::

        results = backtest.run()
        market_df  = results.market_records_df()
        orders_df  = results.orders_df()
        fills_df   = results.fills_df()
        signals_df = results.custom_metrics("signals")
        results.save("/tmp/my_backtest")
    """

    @property
    def PRICE_SCALE(self):
        return Scales.PRICE

    @property
    def SIZE_SCALE(self):
        return Scales.SIZE

    def __init__(self, java_recorder, metadata: BacktestMetadata | None = None):
        self._java = java_recorder
        self._metadata = metadata
        # Keyed by (stream, scaled). Either form is derived from the other, so the two never disagree.
        self._frames: dict[tuple[str, bool], pd.DataFrame] = {}
        self._cached_custom_dfs: dict[str, pd.DataFrame] | None = None

    @property
    def metadata(self) -> BacktestMetadata | None:
        """Backtest metadata (ID, config, dates, etc.). None for legacy results."""
        return self._metadata

    @property
    def backtest_id(self) -> str | None:
        """Unique identifier for this backtest run."""
        return self._metadata.backtest_id if self._metadata else None

    @classmethod
    def from_dataframes(
        cls,
        market_df: pd.DataFrame | None = None,
        orders_df: pd.DataFrame | None = None,
        fills_df: pd.DataFrame | None = None,
        intent_df: pd.DataFrame | None = None,
        custom_metrics: dict[str, pd.DataFrame] | None = None,
        metadata: BacktestMetadata | None = None,
    ) -> "BacktestResults":
        """Construct a BacktestResults directly from DataFrames (no Java recorder).

        Useful for testing and programmatic construction.
        """
        result = cls.__new__(cls)
        result._java = None
        result._metadata = metadata
        frames = (market_df, orders_df, fills_df, intent_df)
        result._frames = {
            (stream, True): df if df is not None else pd.DataFrame() for stream, df in zip(_STREAMS, frames)
        }
        result._cached_custom_dfs = dict(custom_metrics) if custom_metrics else {}
        return result

    @classmethod
    def from_parquet(cls, path: str | Path) -> "BacktestResults":
        """Load backtest results from a directory written by ``save()``.

        Supports local paths and S3 URIs (``s3://bucket/prefix``).

        Args:
            path: Local directory or S3 prefix containing parquet files and metadata.json.

        Returns:
            A ``BacktestResults`` backed by DataFrames read from Parquet, with no live
            Java recorder attached.
        """
        result = cls.__new__(cls)
        result._java = None

        fs, base = resolve_fs(str(path))
        base = base.rstrip("/")

        meta_path = f"{base}/metadata.json"
        result._metadata = None
        if fs_exists(fs, meta_path):
            result._metadata = BacktestMetadata.from_dict(fs_read_json(fs, meta_path))

        result._frames = {(stream, True): fs_read_parquet(fs, f"{base}/{stream}.parquet") for stream in _STREAMS}

        result._cached_custom_dfs = {}
        custom_dir = f"{base}/custom"
        for name, fpath in fs_list_parquets(fs, custom_dir):
            result._cached_custom_dfs[name] = fs_read_parquet(fs, fpath)

        return result

    # ------------------------------------------------------------------
    # Record counts
    # ------------------------------------------------------------------

    @property
    def market_record_count(self) -> int:
        if self._java is None:
            return len(self._frames.get(("market", True), ()))
        return int(self._java.getMarketRecordCount())

    @property
    def order_record_count(self) -> int:
        if self._java is None:
            return len(self._frames.get(("orders", True), ()))
        return int(self._java.getOrderRecordCount())

    @property
    def fill_record_count(self) -> int:
        if self._java is None:
            return len(self._frames.get(("fills", True), ()))
        return int(self._java.getFillRecordCount())

    @property
    def intent_record_count(self) -> int:
        if self._java is None:
            return len(self._frames.get(("intents", True), ()))
        return int(self._java.getIntentRecordCount())

    @property
    def record_depth(self) -> int:
        return int(self._java.getRecordDepth())

    # ------------------------------------------------------------------
    # Market records
    # ------------------------------------------------------------------

    def market_records_df(self, scale_prices: bool = True) -> pd.DataFrame:
        """One row per market data tick.

        Columns always present: timestamp (index), exchange_id, security_id,
        bid_price_0 … bid_price_{D-1}, bid_size_0 … ask_size_{D-1},
        last_trade_price, last_trade_size.

        An empty book side, and the trade columns on a tick with no trade, are NaN.
        """
        return self._frame("market", scale_prices, self._build_market_df)

    def _build_market_df(self) -> pd.DataFrame:
        if self._java is None or self.market_record_count == 0:
            return pd.DataFrame()

        df = _buffer_to_df(self._java.getMarketRecords())
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        prices, sizes = _unit_columns("market", df)
        _null_to_nan(df, prices + sizes)
        return df

    # ------------------------------------------------------------------
    # Order records
    # ------------------------------------------------------------------

    def orders_df(self, scale_prices: bool = True) -> pd.DataFrame:
        """One row per order, written when the order reaches terminal state.

        Columns: submit_timestamp (index), ack_timestamp, terminal_timestamp,
        exchange_id, security_id, strategy_id, client_oid,
        side, order_type, submit_price, submit_size, final_price, final_size,
        modify_count, filled_qty, leaves_qty, avg_fill_price, total_fee, final_status.

        ``final_price`` / ``final_size`` are the order's price and size after its last
        acknowledged modify (the submitted ones if it was never modified).

        ``total_cost`` is the filled notional in price units (1e9 = $1), never rescaled.
        ``avg_fill_price`` is total_cost * SIZE_SCALE / filled_qty (rounded); 0 if the
        order was never filled.
        ``side`` values: "None", "Bid", "Ask".
        ``order_type`` values: "Limit", "Market".
        ``final_status`` values: "Filled", "PartialFill", "Cancelled", "Rejected", "Expired", and
        "Open" for orders still working when the run ended.
        """
        return self._frame("orders", scale_prices, self._build_orders_df)

    def _build_orders_df(self) -> pd.DataFrame:
        if self._java is None or self.order_record_count == 0:
            return pd.DataFrame()

        df = _buffer_to_df(self._java.getOrderRecords())

        # Decode byte-encoded categoricals
        df["side"] = _decode_bytes(df["side"].values, _SIDE_MAP)
        df["order_type"] = _decode_bytes(df["order_type"].values, _OTYPE_MAP)
        df["final_status"] = _decode_bytes(df["final_status"].values, _STATUS_MAP)

        # total_cost is notional in price units, so rescaling by SIZE_SCALE
        # recovers a per-unit price in price units.
        filled_qty = df["filled_qty"].values
        total_cost = df["total_cost"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            df["avg_fill_price"] = np.rint(np.where(
                filled_qty > 0, total_cost * float(self.SIZE_SCALE) / filled_qty, 0
            )).astype(np.int64)

        # A market order has no price; its price columns hold the SBE null.
        _null_to_nan(df, [c for c in ("submit_price", "final_price") if c in df.columns])
        df["submit_timestamp"] = pd.to_datetime(df["submit_timestamp"])
        df["ack_timestamp"] = pd.to_datetime(df["ack_timestamp"])
        df["terminal_timestamp"] = pd.to_datetime(df["terminal_timestamp"])
        return df.set_index("submit_timestamp")

    # ------------------------------------------------------------------
    # Fill records
    # ------------------------------------------------------------------

    def fills_df(self, scale_prices: bool = True) -> pd.DataFrame:
        """One row per fill event (PARTIAL_FILL or FILL).

        Columns: timestamp (index), exchange_id, security_id, strategy_id,
        client_oid, side, fill_price, fill_qty, leaves_qty,
        fee, book_bid_price, book_ask_price.

        ``book_*`` columns reflect the BBO at fill time and can be used to
        compute slippage: ``(fill_price - book_mid_price) / book_mid_price × 10000``
        (sign-adjusted per side). They are NaN when that side of the book was empty
        or no book had been seen yet.
        """
        return self._frame("fills", scale_prices, self._build_fills_df)

    def _build_fills_df(self) -> pd.DataFrame:
        if self._java is None or self.fill_record_count == 0:
            return pd.DataFrame()

        df = _buffer_to_df(self._java.getFillRecords())
        df["side"] = _decode_bytes(df["side"].values, _SIDE_MAP)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        book = ["book_bid_price", "book_ask_price"]
        _null_to_nan(df, book)
        # The recorder leaves these at 0 when no book was seen for the listing before the fill.
        for col in book:
            df[col] = df[col].astype(float).where(df[col] != 0)
        return df

    # ------------------------------------------------------------------
    # Intent records
    # ------------------------------------------------------------------

    def intent_records_df(self, scale_prices: bool = True) -> pd.DataFrame:
        """One row per strategy intent published to the OMS."""
        return self._frame("intents", scale_prices, self._build_intent_df)

    def _build_intent_df(self) -> pd.DataFrame:
        if self._java is None or self.intent_record_count == 0:
            return pd.DataFrame()

        df = _buffer_to_df(self._java.getIntentRecords())
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.set_index("timestamp")

    def _frame(self, stream: str, scaled: bool, build) -> pd.DataFrame:
        key = (stream, scaled)
        if key not in self._frames:
            other = (stream, not scaled)
            if other not in self._frames:
                self._frames[(stream, False)] = build()
            if key not in self._frames:
                self._frames[key] = _rescale(self._frames[other], stream, to_scaled=scaled)
        return self._frames[key]

    # ------------------------------------------------------------------
    # Custom strategy metrics
    # ------------------------------------------------------------------

    def custom_metrics(
        self, name: str | None = None
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Get DataFrames for custom strategy metric streams.

        Args:
            name: If given, return the DataFrame for that named buffer.
                  If None, return a dict mapping buffer name → DataFrame.

        Returns:
            A single DataFrame (if name given) or a dict of all custom DataFrames.
        """
        if self._cached_custom_dfs is None:
            self._cached_custom_dfs = {}
            if self._java is not None:
                for buf in self._java.getCustomBuffers():
                    buf_name = str(buf.getName())
                    self._cached_custom_dfs[buf_name] = _buffer_to_df(buf)

        if name is not None:
            return self._cached_custom_dfs.get(name, pd.DataFrame())
        return dict(self._cached_custom_dfs)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, output: str | Path) -> None:
        """Save all recorder data to Parquet files in the given directory or S3 prefix.

        Writes: market.parquet, orders.parquet, fills.parquet, intents.parquet,
        custom/{name}.parquet for each custom metric stream, and metadata.json.

        Args:
            output: Local directory path or S3 URI (``s3://bucket/prefix``).
        """
        fs, base = resolve_fs(str(output))
        fs_mkdir(fs, base)

        for stream_name, df in [
            ("market", self.market_records_df()),
            ("orders", self.orders_df()),
            ("fills", self.fills_df()),
            ("intents", self.intent_records_df()),
        ]:
            if not df.empty:
                fs_write_parquet(fs, f"{base}/{stream_name}.parquet", df)

        custom = self.custom_metrics()
        if custom:
            custom_dir = f"{base}/custom"
            fs_mkdir(fs, custom_dir)
            for stream_name, df in custom.items():
                if not df.empty:
                    fs_write_parquet(fs, f"{custom_dir}/{stream_name}.parquet", df)

        if self._metadata is not None:
            self._metadata.data_scaled = True
            fs_write_json(fs, f"{base}/metadata.json", self._metadata.to_dict())

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = []
        if self._metadata is not None:
            parts.append(f"id={self._metadata.backtest_id!r}")
        if self._java is not None:
            custom_count = len(list(self._java.getCustomBuffers()))
            parts += [
                f"market_records={self.market_record_count}",
                f"orders={self.order_record_count}",
                f"fills={self.fill_record_count}",
                f"intent_records={self.intent_record_count}",
                f"custom_buffers={custom_count}",
            ]
        else:
            parts.append("from_parquet=True")
        return f"BacktestResults({', '.join(parts)})"
