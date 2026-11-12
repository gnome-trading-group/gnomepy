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
from gnomepy.fees import load_fee_timelines
from gnomepy.java.statics import Scales
from gnomepy.metadata import BacktestMetadata
from gnomepy.registry.api import RegistryClient

# Byte-encoding constants mirrored from BacktestRecorder.java
_SIDE_MAP = {0: "None", 1: "Bid", 2: "Ask"}
_OTYPE_MAP = {0: "Limit", 1: "Market"}
_STATUS_MAP = {0: "Filled", 1: "PartialFill", 2: "Cancelled", 3: "Rejected", 4: "Expired", 5: "Open"}
_LIQUIDITY_MAP = {0: "Unknown", 1: "Maker", 2: "Taker"}
# A program period with no end, as the recorder writes it (Java's Long.MAX_VALUE).
_OPEN_ENDED = np.iinfo(np.int64).max


# SBE null for int64 price and size fields: the recorder writes it for an empty book side or a tick with no trade.
_NULL_LONG = np.iinfo(np.int64).min

_STREAMS = ("market", "orders", "fills", "intents", "rewards")

# (price columns, size columns) per stream. Market columns depend on the recorded depth, so they are matched by name.
_UNIT_COLUMNS = {
    "orders": (
        ("submit_price", "final_price", "avg_fill_price"),
        ("submit_size", "final_size", "filled_qty", "leaves_qty"),
    ),
    "fills": (("fill_price", "book_bid_price", "book_ask_price"), ("fill_qty", "leaves_qty")),
    "intents": (("bid_price", "ask_price", "take_limit_price"), ("bid_size", "ask_size", "take_size")),
    "rewards": (("program_reward", "gross", "market_gross", "market_paid"), ()),
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


class MetricStream:
    """One named stream of metric rows, with its columns declared at runtime.

    Declare the columns, then ``freeze()``; each ``add_*_column`` returns the column's index. Record a row with
    ``begin()``, a ``set_*`` per column, then ``commit()``. A column left unset in a row is 0. Rows go wherever the
    session sends them: the backtest's results, or the session journal in a live or paper session.
    """

    def __init__(self, java_stream):
        self._java = java_stream
        # Bound straight to the Java methods: rows are recorded on the hot path, so no Python wrapper in between.
        self.begin = java_stream.begin
        self.set_long = java_stream.setLong
        self.set_double = java_stream.setDouble
        self.set_int = java_stream.setInt
        self.commit = java_stream.commit

    @property
    def name(self) -> str:
        return str(self._java.name())

    def add_long_column(self, name: str) -> int:
        return int(self._java.addLongColumn(name))

    def add_double_column(self, name: str) -> int:
        return int(self._java.addDoubleColumn(name))

    def add_int_column(self, name: str) -> int:
        return int(self._java.addIntColumn(name))

    def freeze(self) -> None:
        self._java.freeze()


class MetricRecorder:
    """Python wrapper around the Java MetricRecorder a strategy is given.

    Passed to strategies via ``self.metrics`` inside ``register_metrics()``.

    Usage::

        def register_metrics(self):
            stream = self.metrics.create_stream("signals")
            self.ts_col = stream.add_long_column("timestamp")
            self.fv_col = stream.add_double_column("fair_value")
            stream.freeze()
            self.signals = stream

        def on_market_data(self, data):
            self.signals.begin()
            self.signals.set_long(self.ts_col, data.event_timestamp)
            self.signals.set_double(self.fv_col, self.compute_fair_value())
            self.signals.commit()
            return []
    """

    def __init__(self, java_metric_recorder):
        self._java = java_metric_recorder

    def create_stream(self, name: str) -> MetricStream:
        """Declare a new named stream; its name is the parquet file name and DataFrame key it ends up under."""
        return MetricStream(self._java.createStream(name))


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
        rewards_df: pd.DataFrame | None = None,
    ) -> "BacktestResults":
        """Construct a BacktestResults directly from DataFrames (no Java recorder).

        Useful for testing and programmatic construction.
        """
        result = cls.__new__(cls)
        result._java = None
        result._metadata = metadata
        frames = (market_df, orders_df, fills_df, intent_df, rewards_df)
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

        Columns always present: timestamp (index), stream_id, listing_id (the listing the stream carries; NaN for a
        stream of anything else),
        bid_price_0 … bid_price_{D-1}, bid_size_0 … ask_size_{D-1},
        last_trade_price, last_trade_size.

        An empty book side, and the trade columns on a tick with no trade, are NaN.
        """
        return self._frame("market", scale_prices, self._build_market_df)

    def _build_market_df(self) -> pd.DataFrame:
        if self._java is None or self.market_record_count == 0:
            return pd.DataFrame()

        df = _buffer_to_df(self._java.getMarketRecords())
        # The recorder writes 0 for a stream that carries no listing (a vendor feed).
        df["listing_id"] = df["listing_id"].astype("Int64").replace(0, pd.NA)
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
        if "liquidity" in df.columns:
            df["liquidity"] = _decode_bytes(df["liquidity"].values, _LIQUIDITY_MAP)
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

    # ------------------------------------------------------------------
    # Rewards (estimated)
    # ------------------------------------------------------------------

    def rewards_timeseries_df(self, scale_prices: bool = True) -> pd.DataFrame:
        """What each listing's venue liquidity programs had accrued at each snapshot, cumulative. Estimates.

        Columns: timestamp (index), listing_id, venue_program_id, market, period_start, period_end (NaT when the
        program has no end), program_reward, gross (ours on this listing), share_nanos, market_gross (ours on the
        whole market), market_paid (what the venue would pay the market at that gross once the period ends),
        counting_nanos, one_sided_nanos, depth_unknown_nanos, no_data_nanos.
        """
        return self._frame("rewards", scale_prices, self._build_rewards_df)

    def _build_rewards_df(self) -> pd.DataFrame:
        if self._java is None or int(self._java.getRewardRecords().getCount()) == 0:
            return pd.DataFrame()
        df = _buffer_to_df(self._java.getRewardRecords())
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["period_start"] = pd.to_datetime(df["period_start"])
        df["period_end"] = pd.to_datetime(df["period_end"].where(df["period_end"] != _OPEN_ENDED))
        return df.set_index("timestamp")

    def rewards_df(self) -> pd.DataFrame:
        """Estimated venue liquidity rewards, one row per listing and program period, as they stood at the run's end.

        ``gross`` is our estimate in dollars on the listing. A period that ended inside the run is ``closed`` and has
        ``paid``: the venue's closing rule (Kalshi: whole cents, nothing under $1) applied to the whole market, split
        across its listings by gross. An open period's ``paid`` is NaN. The ``*_fraction`` columns divide the period
        into time that counted, a side short of the program's size, a side possibly deeper than the feed shows, and
        no data.
        """
        series = self.rewards_timeseries_df()
        if series.empty:
            return pd.DataFrame()
        run_end = series.index.max()
        last = series.reset_index().groupby(["listing_id", "venue_program_id"], sort=True).last().reset_index()
        last["closed"] = last["period_end"].notna() & (last["period_end"] <= run_end)
        share = (last["gross"] / last["market_gross"]).where(last["market_gross"] > 0, 0.0)
        last["paid"] = (last["market_paid"] * share).where(last["closed"])
        period_ns = (last["period_end"].fillna(run_end) - last["period_start"]).dt.total_seconds() * 1e9
        for kind in ("counting", "one_sided", "depth_unknown", "no_data"):
            last[f"{kind}_fraction"] = (last[f"{kind}_nanos"] / period_ns).where(period_ns > 0)
        columns = ["listing_id", "venue_program_id", "market", "period_start", "period_end", "closed",
                   "program_reward", "gross", "paid", "counting_fraction", "one_sided_fraction",
                   "depth_unknown_fraction", "no_data_fraction"]
        return last[columns]

    def maker_rebates_df(self, registry: RegistryClient | None = None) -> pd.DataFrame:
        """Estimated maker rebates paid apart from the fill (Polymarket International's), one row per MAKER fill.

        A maker fill earns the fee schedule's ``maker_rebate_share`` of the taker fee the same fill would have paid,
        under the schedule in force at the fill as the registry knew it at the run's ``simulation_as_of``. Fills on
        listings without such a rebate are left out. ``rebate`` is in dollars.
        """
        fills = self.fills_df(scale_prices=False)
        if fills.empty or "liquidity" not in fills.columns:
            return pd.DataFrame()
        makers = fills[fills["liquidity"] == "Maker"]
        if makers.empty:
            return pd.DataFrame()
        known_as_of = self._metadata.simulation_as_of if self._metadata is not None else None
        timelines = load_fee_timelines(
            registry or RegistryClient(), sorted(int(i) for i in makers["listing_id"].unique()), known_as_of=known_as_of)
        rows = []
        for ts, fill in makers.iterrows():
            timeline = timelines.get(int(fill["listing_id"]))
            schedule = timeline.at(int(ts.value)) if timeline is not None else None
            if schedule is None or schedule.maker_rebate_share <= 0:
                continue
            taker_fee = schedule.fee(fill["side"] == "Bid", int(fill["fill_price"]), int(fill["fill_qty"]), False)
            rows.append({"timestamp": ts, "listing_id": int(fill["listing_id"]), "client_oid": int(fill["client_oid"]),
                         "share": schedule.maker_rebate_share,
                         "rebate": schedule.maker_rebate_share * taker_fee / Scales.PRICE})
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("timestamp")

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

        Writes: market.parquet, orders.parquet, fills.parquet, intents.parquet, rewards.parquet,
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
            ("rewards", self.rewards_timeseries_df()),
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
