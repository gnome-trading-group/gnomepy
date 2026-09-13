"""Core data layer for the backtest explorer.

Loads BacktestResults into memory and provides windowed, downsampled views
suitable for rendering in the Dash UI without shipping millions of points
to the browser.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from gnomepy.java.recorder import BacktestResults

MAX_CHART_POINTS = 5_000
DEPTH_WINDOW_THRESHOLD_S = 60.0
SLIDER_RESOLUTION = 1_000


@dataclass
class WindowedData:
    market: pd.DataFrame
    fills: pd.DataFrame
    intents: pd.DataFrame
    orders: pd.DataFrame
    custom: dict[str, pd.DataFrame]
    record_depth: int
    is_deep_window: bool


def _lttb(series: pd.Series, n: int) -> pd.Series:
    """Largest-Triangle-Three-Buckets downsampling preserving visual shape."""
    m = len(series)
    if m <= n:
        return series
    y = series.values.astype(float)
    bucket_size = (m - 2) / (n - 2)
    selected = [0]
    a = 0
    for i in range(1, n - 1):
        curr_start = int(i * bucket_size) + 1
        curr_end = min(int((i + 1) * bucket_size) + 1, m)
        next_start = curr_end
        next_end = min(int((i + 2) * bucket_size) + 1, m)
        avg_x = float(next_start + next_end - 1) / 2.0
        avg_y = float(y[next_start:next_end].mean()) if next_start < m else y[-1]
        ax = float(a)
        ay = y[a]
        cx = np.arange(curr_start, curr_end, dtype=float)
        areas = np.abs((ax - avg_x) * (y[curr_start:curr_end] - ay) - (cx - ax) * (avg_y - ay)) * 0.5
        best = curr_start + int(np.argmax(areas))
        selected.append(best)
        a = best
    selected.append(m - 1)
    return series.iloc[selected]


def _lttb_df(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Downsample a DataFrame using LTTB on the mid_price (or first numeric) column."""
    if len(df) <= n:
        return df
    signal_col = "mid_price" if "mid_price" in df.columns else df.select_dtypes("number").columns[0]
    idx = _lttb(df[signal_col], n).index
    return df.loc[idx]


def _detect_record_depth(market_df: pd.DataFrame) -> int:
    return sum(1 for c in market_df.columns if c.startswith("bid_price_")) or 1


def _ensure_mid_price(market_df: pd.DataFrame) -> pd.DataFrame:
    if market_df.empty or "mid_price" in market_df.columns:
        return market_df
    df = market_df.copy()
    bid = df["bid_price_0"].astype(float).where(df["bid_price_0"] > 0)
    ask = df["ask_price_0"].astype(float).where(df["ask_price_0"] > 0)
    df["mid_price"] = ((bid.fillna(ask) + ask.fillna(bid)) / 2.0).ffill()
    return df


def _normalize_custom_dfs(custom_dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Ensure each custom metric DataFrame has a DatetimeIndex."""
    out = {}
    for name, df in custom_dfs.items():
        if df.empty:
            out[name] = df
            continue
        if isinstance(df.index, pd.DatetimeIndex):
            out[name] = df
        elif "timestamp" in df.columns:
            df = df.copy()
            ts = df["timestamp"]
            if pd.api.types.is_integer_dtype(ts):
                df.index = pd.to_datetime(ts)
            else:
                df.index = pd.to_datetime(ts)
            df = df.drop(columns=["timestamp"])
            out[name] = df.sort_index()
        else:
            out[name] = df
    return out


def _slice(df: pd.DataFrame, t_start: pd.Timestamp, t_end: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    return df.loc[t_start:t_end]


def _slice_orders(orders_df: pd.DataFrame, t_start: pd.Timestamp, t_end: pd.Timestamp) -> pd.DataFrame:
    if orders_df.empty:
        return orders_df
    mask = (orders_df.index >= t_start) & (orders_df.index <= t_end)
    return orders_df[mask]


def _safe_float(val) -> float:
    try:
        v = float(val)
        return v if v == v else 0.0
    except (TypeError, ValueError):
        return 0.0


def _compute_fill_slippage(fills: pd.DataFrame) -> pd.DataFrame:
    if fills.empty or "book_bid_price" not in fills.columns:
        return fills
    fills = fills.copy()
    mid = (fills["book_bid_price"].astype(float) + fills["book_ask_price"].astype(float)) / 2.0
    is_bid = fills["side"].str.upper().str.contains("BID")
    signed_slip = (fills["fill_price"].astype(float) - mid) * is_bid.map({True: 1.0, False: -1.0})
    fills["slippage_bps"] = (signed_slip / mid * 10_000).round(2)
    fills["slippage_usd"] = signed_slip * fills["fill_qty"].astype(float)
    return fills


class ExplorerDataStore:
    """Holds all data for a single backtest and serves windowed views."""

    def __init__(self, results: BacktestResults, label: str = "A", _progress=None):
        self.label = label
        self.metadata = results._metadata
        _p = _progress or (lambda msg: None)

        self.market_df = _ensure_mid_price(results.market_records_df())
        self.fills_df = _compute_fill_slippage(results.fills_df())
        self.orders_df = results.orders_df()
        self.intents_df = results.intent_records_df()
        self.custom_dfs = _normalize_custom_dfs(results.custom_metrics())

        self.record_depth = _detect_record_depth(self.market_df)

        _p(f"computing PnL curves ({len(self.market_df):,} market rows)…")
        from gnomepy.reporting.metrics import build_curves
        self.curves = build_curves(self.market_df, self.fills_df)

        if not self.market_df.empty:
            self.t_min: pd.Timestamp = self.market_df.index[0]
            self.t_max: pd.Timestamp = self.market_df.index[-1]
        else:
            self.t_min = pd.Timestamp.min
            self.t_max = pd.Timestamp.max

        self._event_ts_by_type = self._build_event_index()
        _p("resolving listing labels…")
        self._listing_labels: dict[tuple[int, int], str] = self._resolve_listing_labels()

    def _build_event_index(self) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        if not self.fills_df.empty:
            result["fill"] = np.sort(self.fills_df.index.values.astype(np.int64))
        if not self.intents_df.empty:
            result["intent"] = np.sort(self.intents_df.index.values.astype(np.int64))
        if not self.orders_df.empty:
            submits = self.orders_df.index.values.astype(np.int64)
            terminals = pd.to_datetime(self.orders_df["terminal_timestamp"]).values.astype(np.int64)
            result["order"] = np.sort(np.concatenate([submits, terminals]))
        all_ts: list[np.ndarray] = list(result.values())
        if all_ts:
            result["all"] = np.sort(np.concatenate(all_ts))
        return result

    def event_timestamps(self, event_types: list[str] | None = None) -> np.ndarray:
        if not event_types or "all" in event_types:
            return self._event_ts_by_type.get("all", np.array([], dtype=np.int64))
        arrays = [self._event_ts_by_type[t] for t in event_types if t in self._event_ts_by_type]
        if not arrays:
            return np.array([], dtype=np.int64)
        return np.sort(np.concatenate(arrays))

    def listings(self) -> list[tuple[int, int]]:
        """Return sorted unique (exchange_id, security_id) pairs."""
        if self.market_df.empty:
            return []
        return sorted(
            self.market_df.groupby(["exchange_id", "security_id"], sort=False).groups.keys()
        )

    def listing_label(self, eid: int, sid: int) -> str:
        """Human-readable label for a listing, falling back to eid/sid."""
        label = self._listing_labels.get((eid, sid))
        return label if label else f"{eid}/{sid}"

    def _resolve_listing_labels(self) -> dict[tuple[int, int], str]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        labels: dict[tuple[int, int], str] = {}
        config = getattr(self.metadata, "config", None) if self.metadata else None
        if not config:
            return labels
        raw_listings = config.get("listings")
        if not raw_listings:
            return labels
        listing_ids: list[int] = []
        for entry in raw_listings:
            if isinstance(entry, dict):
                lid = entry.get("listing_id")
                if lid is not None:
                    listing_ids.append(int(lid))
            elif isinstance(entry, int):
                listing_ids.append(entry)
        if not listing_ids:
            return labels

        def _fetch(lid: int) -> tuple[tuple[int, int], str] | None:
            try:
                from gnomepy.registry.api import RegistryClient
                client = RegistryClient()
                results = client.get_listing(listing_id=lid)
                if not results:
                    return None
                listing = results[0]
                sec_results = client.get_security(security_id=listing.security_id)
                exch_results = client.get_exchange(exchange_id=listing.exchange_id)
                sym = sec_results[0].symbol if sec_results else f"sid={listing.security_id}"
                exch_name = exch_results[0].exchange_name if exch_results else f"eid={listing.exchange_id}"
                return (listing.exchange_id, listing.security_id), f"{sym} @ {exch_name}"
            except Exception:
                return None

        try:
            with ThreadPoolExecutor(max_workers=min(len(listing_ids), 8)) as pool:
                for result in as_completed([pool.submit(_fetch, lid) for lid in listing_ids]):
                    entry = result.result()
                    if entry:
                        labels[entry[0]] = entry[1]
        except Exception:
            pass
        return labels

    def window(
        self,
        t_start: pd.Timestamp,
        t_end: pd.Timestamp,
        listing: tuple[int, int] | None = None,
        max_points: int = MAX_CHART_POINTS,
    ) -> WindowedData:
        window_seconds = (t_end - t_start).total_seconds()
        is_deep = window_seconds < DEPTH_WINDOW_THRESHOLD_S and self.record_depth > 1

        mkt = _slice(self.market_df, t_start, t_end)
        fills = _slice(self.fills_df, t_start, t_end)
        intents = _slice(self.intents_df, t_start, t_end)
        orders = _slice_orders(self.orders_df, t_start, t_end)

        if listing is not None:
            eid, sid = listing
            if not mkt.empty:
                mkt = mkt[(mkt["exchange_id"] == eid) & (mkt["security_id"] == sid)]
            if not fills.empty and "exchange_id" in fills.columns:
                fills = fills[(fills["exchange_id"] == eid) & (fills["security_id"] == sid)]
            if not orders.empty and "exchange_id" in orders.columns:
                orders = orders[(orders["exchange_id"] == eid) & (orders["security_id"] == sid)]
            if not intents.empty and "exchange_id" in intents.columns:
                intents = intents[(intents["exchange_id"] == eid) & (intents["security_id"] == sid)]

        if len(mkt) > max_points:
            mkt = _lttb_df(mkt, max_points)

        return WindowedData(
            market=mkt,
            fills=fills,
            intents=intents,
            orders=orders,
            custom={name: _slice(df, t_start, t_end) for name, df in self.custom_dfs.items()},
            record_depth=self.record_depth,
            is_deep_window=is_deep,
        )

    def slider_to_ts(self, value: float) -> pd.Timestamp:
        span_ns = (self.t_max - self.t_min).value
        offset_ns = int(span_ns * value / SLIDER_RESOLUTION)
        return self.t_min + pd.Timedelta(offset_ns, unit="ns")

    def ts_to_slider(self, ts: pd.Timestamp) -> float:
        span_ns = (self.t_max - self.t_min).value
        if span_ns == 0:
            return 0.0
        offset_ns = (ts - self.t_min).value
        return max(0.0, min(float(SLIDER_RESOLUTION), offset_ns / span_ns * SLIDER_RESOLUTION))

    def summary_label(self) -> str:
        if self.metadata is None:
            return self.label
        bid = self.metadata.backtest_id or self.label
        strategy = self.metadata.strategy or ""
        if strategy and ":" in strategy:
            strategy = strategy.split(":")[-1]
        return f"{bid} — {strategy}" if strategy else bid

    def book_snapshot(
        self,
        timestamp: pd.Timestamp,
        listing: tuple[int, int] | None = None,
    ) -> dict[tuple[int, int], dict]:
        result: dict[tuple[int, int], dict] = {}
        mkt = self.market_df
        if mkt.empty:
            return result
        ts_ns = timestamp.value
        listings_to_check = [listing] if listing else self.listings()
        for eid, sid in listings_to_check:
            df = mkt[(mkt["exchange_id"] == eid) & (mkt["security_id"] == sid)]
            if df.empty:
                continue
            idx = int(np.searchsorted(df.index.values.astype(np.int64), ts_ns, side="right")) - 1
            if idx < 0:
                continue
            row = df.iloc[idx]
            bids: list[tuple[float, float]] = []
            asks: list[tuple[float, float]] = []
            for lvl in range(self.record_depth):
                bp = _safe_float(row.get(f"bid_price_{lvl}", 0))
                bs = _safe_float(row.get(f"bid_size_{lvl}", 0))
                ap = _safe_float(row.get(f"ask_price_{lvl}", 0))
                as_ = _safe_float(row.get(f"ask_size_{lvl}", 0))
                if bp > 0:
                    bids.append((bp, bs))
                if ap > 0:
                    asks.append((ap, as_))
            result[(eid, sid)] = {"bids": bids, "asks": asks, "timestamp": df.index[idx]}
        return result

    def intent_at(
        self,
        timestamp: pd.Timestamp,
        listing: tuple[int, int] | None = None,
    ) -> dict[tuple[int, int], dict]:
        result: dict[tuple[int, int], dict] = {}
        if self.intents_df.empty:
            return result
        ts_ns = timestamp.value
        listings_to_check = [listing] if listing else self.listings()
        for eid, sid in listings_to_check:
            df = self.intents_df
            if "exchange_id" in df.columns:
                df = df[(df["exchange_id"] == eid) & (df["security_id"] == sid)]
            if df.empty:
                continue
            idx = int(np.searchsorted(df.index.values.astype(np.int64), ts_ns, side="right")) - 1
            if idx < 0:
                continue
            row = df.iloc[idx]
            result[(eid, sid)] = {
                "bid_price": _safe_float(row.get("bid_price", 0)),
                "bid_size": _safe_float(row.get("bid_size", 0)),
                "ask_price": _safe_float(row.get("ask_price", 0)),
                "ask_size": _safe_float(row.get("ask_size", 0)),
                "take_limit_price": _safe_float(row.get("take_limit_price", 0)),
                "take_side": str(row.get("take_side", "") or ""),
                "take_size": _safe_float(row.get("take_size", 0)),
            }
        return result

    def fills_near(
        self,
        timestamp: pd.Timestamp,
        window_ns: int = 500_000_000,
        listing: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        if self.fills_df.empty:
            return self.fills_df
        delta = pd.Timedelta(window_ns, unit="ns")
        t_lo = timestamp - delta
        t_hi = timestamp + delta
        fills = _slice(self.fills_df, t_lo, t_hi)
        if listing is not None and not fills.empty and "exchange_id" in fills.columns:
            eid, sid = listing
            fills = fills[(fills["exchange_id"] == eid) & (fills["security_id"] == sid)]
        return fills


class ComparisonStore:
    """Wraps two ExplorerDataStore instances for synchronized comparison views."""

    def __init__(self, a: ExplorerDataStore, b: ExplorerDataStore):
        self.a = a
        self.b = b
        self.same_dates = self._detect_same_dates()
        self.t_min = min(a.t_min, b.t_min)
        self.t_max = max(a.t_max, b.t_max)

    def _detect_same_dates(self) -> bool:
        if self.a.metadata is None or self.b.metadata is None:
            return False
        return self.a.metadata.start_date == self.b.metadata.start_date

    def window(self, t_start: pd.Timestamp, t_end: pd.Timestamp) -> tuple[WindowedData, WindowedData]:
        return self.a.window(t_start, t_end), self.b.window(t_start, t_end)

    def pnl_delta(self, t_start: pd.Timestamp, t_end: pd.Timestamp) -> pd.Series:
        """PnL of A minus PnL of B over the window, forward-filled and aligned."""
        pnl_a = _slice(self.a.curves.pnl, t_start, t_end)
        pnl_b = _slice(self.b.curves.pnl, t_start, t_end)
        if pnl_a.empty or pnl_b.empty:
            return pd.Series(dtype=float)
        combined_idx = pnl_a.index.union(pnl_b.index)
        a_aligned = pnl_a.reindex(combined_idx).ffill()
        b_aligned = pnl_b.reindex(combined_idx).ffill()
        return (a_aligned - b_aligned).dropna()

    def config_diff(self) -> dict[str, tuple]:
        """Return dict of {key: (value_a, value_b)} for differing strategy args."""
        args_a = self.a.metadata.strategy_args or {} if self.a.metadata else {}
        args_b = self.b.metadata.strategy_args or {} if self.b.metadata else {}
        all_keys = set(args_a) | set(args_b)
        diffs = {}
        for k in sorted(all_keys):
            va, vb = args_a.get(k), args_b.get(k)
            if va != vb:
                diffs[k] = (va, vb)
        return diffs

    def divergence_windows(self, bin_seconds: float = 1.0) -> pd.DataFrame:
        """Return time bins where one run traded but the other did not."""
        if self.a.fills_df.empty and self.b.fills_df.empty:
            return pd.DataFrame()
        freq = f"{bin_seconds}s"
        bins_a = set()
        bins_b = set()
        if not self.a.fills_df.empty:
            bins_a = set(self.a.fills_df.index.floor(freq))
        if not self.b.fills_df.empty:
            bins_b = set(self.b.fills_df.index.floor(freq))
        a_only = sorted(bins_a - bins_b)
        b_only = sorted(bins_b - bins_a)
        rows = [{"timestamp": ts, "source": "A"} for ts in a_only]
        rows += [{"timestamp": ts, "source": "B"} for ts in b_only]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
