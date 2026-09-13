"""Plotly chart builders for BacktestReport."""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from gnomepy.config import config as gnome_config
from gnomepy.registry.api import RegistryClient
from gnomepy.registry.types import Exchange, Listing, Security, SecurityType
from gnomepy.reporting.metrics import _is_buy

if TYPE_CHECKING:
    from gnomepy.reporting.report import BacktestReport


@dataclass
class ReportSection:
    """A configurable section of the HTML report.

    ``render`` should accept a ``BacktestReport`` and return:
    - ``str`` — raw HTML inserted inside a ``<section>`` wrapper
    - ``go.Figure`` — auto-converted to embedded plotly HTML
    - ``None`` — section is silently skipped
    """

    name: str
    title: str
    render: Callable[..., str | go.Figure | None]

pio.templates.default = "ggplot2"

DEFAULT_MAX_POINTS = 50_000


_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

_LEGEND_LAYOUT = dict(
    orientation="v",
    yanchor="top",
    y=1,
    xanchor="left",
    x=1.02,
    bgcolor="rgba(255,255,255,0.8)",
    bordercolor="rgba(0,0,0,0.1)",
    borderwidth=1,
)


def _with_alpha(hex_color: str, alpha: float = 0.5) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _iter_symbols(market_df: pd.DataFrame) -> list[tuple[int, int]]:
    """Get unique (exchange_id, security_id) pairs from market data."""
    if market_df.empty:
        return []
    return list(
        market_df.groupby(["exchange_id", "security_id"], sort=False).ngroups
        and market_df.groupby(["exchange_id", "security_id"], sort=False).groups.keys()
    )


def _sym_label(eid: int, sid: int, lid: int | None = None) -> str:
    if lid is not None:
        return f"{eid}/{sid}/{lid}"
    return f"{eid}/{sid}"


@dataclass
class _ListingContext:
    eid_sid_to_lid: dict[tuple[int, int], int]
    listings: dict[int, Listing]
    securities: dict[int, Security]
    exchanges: dict[int, Exchange]
    controller_ui: str


def _get_listing_context(report: "BacktestReport") -> "_ListingContext":
    cached = getattr(report, "_listing_ctx", None)
    if cached is not None:
        return cached

    eid_sid_to_lid: dict[tuple[int, int], int] = {}
    listings_map: dict[int, Listing] = {}
    securities_map: dict[int, Security] = {}
    exchanges_map: dict[int, Exchange] = {}

    config = getattr(report, "_config", None)
    if config:
        raw_listings = config.get("listings")
        if raw_listings:
            listing_ids: list[int] = []
            for entry in raw_listings:
                if isinstance(entry, dict):
                    lid = entry.get("listing_id")
                    if lid is not None:
                        listing_ids.append(int(lid))
                elif isinstance(entry, int):
                    listing_ids.append(entry)

            if listing_ids:
                try:
                    client = RegistryClient()
                    for lid in listing_ids:
                        results = client.get_listing(listing_id=lid)
                        if results:
                            listing = results[0]
                            listings_map[lid] = listing
                            eid_sid_to_lid[(listing.exchange_id, listing.security_id)] = lid

                    for sid in {l.security_id for l in listings_map.values()}:
                        results = client.get_security(security_id=sid)
                        if results:
                            securities_map[sid] = results[0]

                    for eid in {l.exchange_id for l in listings_map.values()}:
                        results = client.get_exchange(exchange_id=eid)
                        if results:
                            exchanges_map[eid] = results[0]
                except Exception:
                    pass

    ctx = _ListingContext(
        eid_sid_to_lid=eid_sid_to_lid,
        listings=listings_map,
        securities=securities_map,
        exchanges=exchanges_map,
        controller_ui=gnome_config.CONTROLLER_UI_URL,
    )
    report._listing_ctx = ctx
    return ctx


def _build_color_map(report: "BacktestReport") -> dict[tuple[int, int], str]:
    """Assign a deterministic color to each (exchange_id, security_id) pair."""
    cached = getattr(report, "_color_map", None)
    if cached is not None:
        return cached
    pairs: set[tuple[int, int]] = set()
    if not report._market_df.empty:
        for pair in report._market_df.groupby(["exchange_id", "security_id"], sort=False).groups:
            pairs.add(pair)
    fills = report.fills
    if not fills.empty and "exchange_id" in fills.columns and "security_id" in fills.columns:
        for pair in fills.groupby(["exchange_id", "security_id"], sort=False).groups:
            pairs.add(pair)
    color_map = {pair: _PALETTE[i % len(_PALETTE)] for i, pair in enumerate(sorted(pairs))}
    report._color_map = color_map
    return color_map


def _lttb(series: pd.Series, n: int) -> pd.Series:
    """Largest-Triangle-Three-Buckets downsampling — preserves visual shape."""
    import numpy as np

    m = len(series)
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
        areas = np.abs((ax - avg_x) * (y[curr_start:curr_end] - ay)
                       - (cx - ax) * (avg_y - ay)) * 0.5
        best = curr_start + int(np.argmax(areas))
        selected.append(best)
        a = best

    selected.append(m - 1)
    return series.iloc[selected]


def _downsample(series: pd.Series, max_points: int | None = DEFAULT_MAX_POINTS) -> pd.Series:
    """Downsample a Series to at most *max_points* using LTTB. Pass ``None`` to disable."""
    if max_points is None or len(series) <= max_points:
        return series
    return _lttb(series, max_points)


def plot_pnl(
    report: "BacktestReport",
    *,
    show_mid: bool = True,
    show_fills: bool = True,
    title: str | None = None,
    max_points: int | None = DEFAULT_MAX_POINTS,
) -> go.Figure:
    """PnL + drawdown chart with optional mid-price overlay and fill markers."""
    pnl = _downsample(report.pnl_curve, max_points)
    peak = pnl.cummax()
    dd = pnl - peak

    specs = [[{"secondary_y": show_mid}], [{"secondary_y": False}]]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
        vertical_spacing=0.04,
        subplot_titles=("Mark-to-market PnL", "Drawdown"),
        specs=specs,
    )

    if show_mid and not report._market_df.empty:
        symbols = _iter_symbols(report._market_df)
        ctx = _get_listing_context(report)
        color_map = _build_color_map(report)
        for i, (eid, sid) in enumerate(symbols):
            sym_mkt = report._market_df[
                (report._market_df["exchange_id"] == eid)
                & (report._market_df["security_id"] == sid)
            ].sort_index()
            sym_mkt = sym_mkt[~sym_mkt.index.duplicated(keep="last")]
            mid = _downsample(sym_mkt["mid_price"].astype(float), max_points)
            color = _with_alpha(color_map.get((eid, sid), _PALETTE[i % len(_PALETTE)]))
            lid = ctx.eid_sid_to_lid.get((eid, sid))
            name = "mid price" if len(symbols) == 1 else f"mid {_sym_label(eid, sid, lid)}"
            fig.add_trace(
                go.Scattergl(
                    x=mid.index, y=mid.values, mode="lines", name=name,
                    line=dict(width=0.8, color=color),
                    hovertemplate="%{y:,.4f}<extra>%{fullData.name}</extra>",
                ),
                row=1, col=1, secondary_y=True,
            )

    fig.add_trace(
        go.Scattergl(x=pnl.index, y=pnl.values, mode="lines", name="PnL",
                     line=dict(width=1.2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown",
                   line=dict(width=1, color="crimson"), fill="tozeroy",
                   fillcolor="rgba(220,20,60,0.2)"),
        row=2, col=1,
    )

    fills = report.fills
    if show_fills and not fills.empty:
        pnl_df = pd.DataFrame({"timestamp": pnl.index, "pnl_val": pnl.values})
        fills_df = pd.DataFrame({"timestamp": fills.index}).reset_index(drop=True)
        eq_merged = pd.merge_asof(
            fills_df.sort_values("timestamp"),
            pnl_df.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        eq_at_fills = eq_merged["pnl_val"].values
        buy_mask = fills["side"].map(_is_buy).values
        fig.add_trace(
            go.Scattergl(
                x=fills.index[buy_mask], y=eq_at_fills[buy_mask], mode="markers",
                name="buy",
                marker=dict(symbol="triangle-up", size=7, color="#2ca02c"),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=fills.index[~buy_mask], y=eq_at_fills[~buy_mask], mode="markers",
                name="sell",
                marker=dict(symbol="triangle-down", size=7, color="#d62728"),
            ),
            row=1, col=1,
        )

    fig.update_layout(
        height=600,
        title=title or "Mark-to-market PnL",
        hovermode="x unified",
        legend=_LEGEND_LAYOUT,
        margin=dict(r=160),
    )
    fig.update_yaxes(title_text="PnL", row=1, col=1, secondary_y=False)
    if show_mid:
        fig.update_yaxes(title_text="mid", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="DD", row=2, col=1)
    fig.update_xaxes(title_text="time", row=2, col=1)
    return fig


def plot_position(
    report: "BacktestReport",
    *,
    title: str | None = None,
    max_points: int | None = DEFAULT_MAX_POINTS,
) -> go.Figure:
    """Position, cumulative fees, and cumulative volume subplots."""
    pos_by_sym = report.position_by_symbol
    fees = _downsample(report.fees_curve, max_points)
    vol = _downsample(report.volume_curve, max_points)

    multi = len(pos_by_sym.columns) > 1

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.4, 0.3, 0.3],
        vertical_spacing=0.04,
        subplot_titles=("Position", "Cumulative fees", "Cumulative volume"),
    )

    if multi:
        ctx = _get_listing_context(report)
        color_map = _build_color_map(report)
        for i, col in enumerate(pos_by_sym.columns):
            if isinstance(col, tuple):
                lid = ctx.eid_sid_to_lid.get(col)
                label = _sym_label(*col, lid)
                color = color_map.get(col, _PALETTE[i % len(_PALETTE)])
            else:
                label = str(col)
                color = _PALETTE[i % len(_PALETTE)]
            s = _downsample(pos_by_sym[col], max_points)
            fig.add_trace(
                go.Scattergl(
                    x=s.index, y=s.values, mode="lines", name=label,
                    line=dict(width=1, shape="hv", color=color),
                ),
                row=1, col=1,
            )
        total = _downsample(report.position_curve, max_points)
        fig.add_trace(
            go.Scattergl(
                x=total.index, y=total.values, mode="lines", name="total",
                line=dict(width=1.5, shape="hv", color="#333", dash="dash"),
            ),
            row=1, col=1,
        )
    else:
        pos = _downsample(report.position_curve, max_points)
        fig.add_trace(
            go.Scatter(
                x=pos.index, y=pos.values, mode="lines", name="position",
                line=dict(width=1, shape="hv", color="#1f77b4"),
                fill="tozeroy", fillcolor="rgba(31,119,180,0.2)",
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Scattergl(
            x=fees.index, y=fees.values, mode="lines", name="fees",
            line=dict(width=1, color="#ff7f0e"),
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=vol.index, y=vol.values, mode="lines", name="volume",
            line=dict(width=1, color="#9467bd"),
        ),
        row=3, col=1,
    )
    fig.update_layout(
        height=650, hovermode="x unified",
        showlegend=multi,
        legend=_LEGEND_LAYOUT if multi else None,
        margin=dict(r=160) if multi else None,
        title=title or "Position / Fees / Volume",
    )
    return fig


def plot_pnl_by_symbol(
    report: "BacktestReport",
    *,
    title: str | None = None,
    max_points: int | None = DEFAULT_MAX_POINTS,
) -> go.Figure:
    """Per-symbol PnL curves on a single chart."""
    by_sym = report.pnl_by_symbol
    ctx = _get_listing_context(report)
    color_map = _build_color_map(report)
    fig = go.Figure()
    for i, col in enumerate(by_sym.columns):
        if isinstance(col, tuple):
            lid = ctx.eid_sid_to_lid.get(col)
            label = _sym_label(col[0], col[1], lid)
            color = color_map.get(col, _PALETTE[i % len(_PALETTE)])
        else:
            label = str(col)
            color = _PALETTE[i % len(_PALETTE)]
        s = _downsample(by_sym[col], max_points)
        fig.add_trace(
            go.Scattergl(
                x=s.index, y=s.values, mode="lines",
                name=label, line=dict(width=1.2, color=color),
            ),
        )
    fig.update_layout(
        height=450,
        title=title or "PnL by symbol",
        hovermode="x unified",
        yaxis_title="PnL",
        xaxis_title="time",
        legend=_LEGEND_LAYOUT,
        margin=dict(r=160),
    )
    return fig


def _fmt(key: str, value) -> str:
    """Format a metric value using _METRIC_FORMAT or a sensible default."""
    fmt = _METRIC_FORMAT.get(key)
    if fmt:
        return f"{value:{fmt}}"
    if isinstance(value, float):
        return f"{value:,.4f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _signed_color(value: float) -> str:
    return "#16a34a" if value >= 0 else "#dc2626"


_KPI_KEYS = [
    ("final_pnl", "Final PnL", True),
    ("sharpe", "Sharpe", True),
    ("sortino", "Sortino", True),
    ("total_fees", "Total Fees", True),
    ("fill_count", "Fills", False),
    ("final_position", "Final Position", False),
    ("total_volume", "Total Volume", False),
    ("total_notional", "Notional Volume", False),
    ("duration_seconds", "Duration", False),
]


def _kpi_cards_html(summary: dict) -> str:
    """Render key metrics as styled cards."""
    cards = []
    for key, label, colored in _KPI_KEYS:
        value = summary.get(key, 0)
        formatted = _fmt(key, value)
        if key == "duration_seconds":
            formatted += "s"
        color = _signed_color(value) if colored else "#334155"
        cards.append((label, formatted, color))

    items = []
    for label, value, color in cards:
        items.append(
            f'<div class="kpi-card">'
            f'<div class="kpi-value" style="color:{color}">{value}</div>'
            f'<div class="kpi-label">{label}</div>'
            f'</div>'
        )
    return '<div class="kpi-row">' + "".join(items) + '</div>'


def plot_spread(
    report: "BacktestReport",
    *,
    title: str | None = None,
    max_points: int | None = DEFAULT_MAX_POINTS,
) -> go.Figure:
    """Bid-ask spread over time, one line per symbol."""
    mkt = report._market_df.sort_index()
    symbols = _iter_symbols(mkt)
    multi = len(symbols) > 1

    ctx = _get_listing_context(report)
    color_map = _build_color_map(report)
    fig = go.Figure()
    for i, (eid, sid) in enumerate(symbols):
        sym_mkt = mkt[(mkt["exchange_id"] == eid) & (mkt["security_id"] == sid)]
        if "spread" in sym_mkt.columns:
            spread = _downsample(sym_mkt["spread"].astype(float), max_points)
        else:
            spread = _downsample(
                (sym_mkt["ask_price_0"] - sym_mkt["bid_price_0"]).astype(float),
                max_points,
            )
        lid = ctx.eid_sid_to_lid.get((eid, sid))
        name = f"spread {_sym_label(eid, sid, lid)}" if multi else "spread"
        color = color_map.get((eid, sid), _PALETTE[i % len(_PALETTE)])
        fig.add_trace(
            go.Scattergl(
                x=spread.index, y=spread.values, mode="lines", name=name,
                line=dict(width=0.8, color=color),
            ),
        )

    fig.update_layout(
        height=350,
        title=title or "Bid-Ask Spread",
        hovermode="x unified",
        yaxis_title="spread",
        xaxis_title="time",
        showlegend=multi,
        legend=_LEGEND_LAYOUT if multi else None,
        margin=dict(r=160) if multi else None,
    )
    return fig


def plot_cross_exchange_spread(
    report: "BacktestReport",
    exchange_id_a: int | None = None,
    exchange_id_b: int | None = None,
    security_id: int | None = None,
    *,
    title: str | None = None,
    max_points: int | None = DEFAULT_MAX_POINTS,
) -> go.Figure:
    """Spread between two exchanges for the same security (in bps)."""
    mkt = report._market_df.sort_index()
    symbols = _iter_symbols(mkt)

    if exchange_id_a is None or exchange_id_b is None:
        exchange_ids = sorted(set(eid for eid, _ in symbols))
        if len(exchange_ids) < 2:
            fig = go.Figure()
            fig.update_layout(title="Cross-Exchange Spread (need 2+ exchanges)", height=350)
            return fig
        exchange_id_a = exchange_ids[0]
        exchange_id_b = exchange_ids[1]
    if security_id is None:
        security_id = symbols[0][1] if symbols else 1

    mkt_a = mkt[(mkt["exchange_id"] == exchange_id_a) & (mkt["security_id"] == security_id)]
    mkt_b = mkt[(mkt["exchange_id"] == exchange_id_b) & (mkt["security_id"] == security_id)]

    if mkt_a.empty or mkt_b.empty:
        fig = go.Figure()
        fig.update_layout(title="Cross-Exchange Spread (missing data)", height=350)
        return fig

    a_df = pd.DataFrame({"timestamp": mkt_a.index, "mid_a": mkt_a["mid_price"].astype(float).values}).reset_index(drop=True)
    b_df = pd.DataFrame({"timestamp": mkt_b.index, "mid_b": mkt_b["mid_price"].astype(float).values}).reset_index(drop=True)

    merged = pd.merge_asof(
        a_df.sort_values("timestamp"),
        b_df.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    ).dropna()

    avg_mid = (merged["mid_a"] + merged["mid_b"]) / 2
    valid = avg_mid > 0
    spread_bps = pd.Series(
        ((merged["mid_b"] - merged["mid_a"]) / avg_mid * 10_000).values,
        index=pd.DatetimeIndex(merged["timestamp"].values),
    )[valid.values]

    spread_bps = _downsample(spread_bps, max_points)

    ctx = _get_listing_context(report)
    lid_a = ctx.eid_sid_to_lid.get((exchange_id_a, security_id))
    lid_b = ctx.eid_sid_to_lid.get((exchange_id_b, security_id))
    label_a = _sym_label(exchange_id_a, security_id, lid_a)
    label_b = _sym_label(exchange_id_b, security_id, lid_b)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=spread_bps.index, y=spread_bps.values, mode="lines",
            name=f"{label_b} - {label_a}",
            line=dict(width=1, color="#6366f1"),
        ),
    )
    fig.add_hline(y=0, line_dash="dot", line_color="grey", opacity=0.4)

    fig.update_layout(
        height=350,
        title=title or f"Cross-Exchange Spread ({label_a} vs {label_b}, bps)",
        hovermode="x unified",
        yaxis_title="spread (bps)",
        xaxis_title="time",
    )
    return fig


MAX_FILLS_DISPLAY = 500
MAX_WARNINGS_DISPLAY = 500


def _fills_table_html(report: "BacktestReport") -> str:
    """Render fills as a scrollable HTML table, capped at MAX_FILLS_DISPLAY rows."""
    import html as _html

    fills = report.fills
    if fills.empty:
        return '<p class="muted">No fills.</p>'

    total = len(fills)
    truncated = total > MAX_FILLS_DISPLAY
    df = fills.head(MAX_FILLS_DISPLAY) if truncated else fills

    cols = ["side", "fill_price", "fill_qty", "leaves_qty", "fee", "book_bid_price", "book_ask_price"]
    available = [c for c in cols if c in df.columns]
    display = df[available].copy()

    # Join order details (order_type, submit_price) via client_oid
    if report._results is not None and "client_oid" in df.columns:
        try:
            orders = report._results.orders_df()
            if not orders.empty and "client_oid" in orders.columns:
                deduped = orders.drop_duplicates(subset="client_oid").set_index("client_oid")
                oids = df["client_oid"].values
                display["order_type"] = pd.array(deduped["order_type"].reindex(oids).values)
                display["submit_price"] = pd.array(deduped["submit_price"].reindex(oids).values)
        except Exception:
            pass

    # Listing ID column with tooltip
    ctx = _get_listing_context(report)
    if ctx.eid_sid_to_lid and "exchange_id" in df.columns and "security_id" in df.columns:
        def _lid_cell(row):
            lid = ctx.eid_sid_to_lid.get((row["exchange_id"], row["security_id"]))
            if lid is None:
                return ""
            listing = ctx.listings.get(lid)
            if listing:
                sec = ctx.securities.get(listing.security_id)
                exch = ctx.exchanges.get(listing.exchange_id)
                sym = _html.escape(sec.symbol) if sec else "?"
                exch_name = _html.escape(exch.exchange_name) if exch else "?"
                tooltip = f"{sym} @ {exch_name}"
                return f'<span title="{tooltip}">{lid}</span>'
            return str(lid)

        display.insert(0, "listing_id", df.apply(_lid_cell, axis=1).values)

    rename_map = {
        "listing_id": "Listing",
        "side": "Side",
        "fill_price": "Fill Price",
        "fill_qty": "Fill Qty",
        "leaves_qty": "Leaves Qty",
        "fee": "Fee",
        "book_bid_price": "Book Bid",
        "book_ask_price": "Book Ask",
        "order_type": "Order Type",
        "submit_price": "Submit Price",
    }
    display = display.rename(columns={k: v for k, v in rename_map.items() if k in display.columns})

    display.index = display.index.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]
    display.index.name = "Timestamp"

    table_html = display.to_html(
        classes="fills-table",
        float_format=lambda x: f"{x:,.6f}",
        escape=False,
    )

    # Color side cells
    table_html = table_html.replace(
        "<td>Bid</td>",
        '<td style="color:#16a34a;font-weight:600">Bid</td>',
    ).replace(
        "<td>Ask</td>",
        '<td style="color:#dc2626;font-weight:600">Ask</td>',
    )

    note = ""
    if truncated:
        note = f'<p class="muted">Showing first {MAX_FILLS_DISPLAY:,} of {total:,} fills.</p>'
    return f'<div class="fills-scroll">{table_html}</div>{note}'


_METRIC_FORMAT: dict[str, str] = {
    "duration_seconds": ",.1f",
    "market_record_count": ",",
    "intent_record_count": ",",
    "fill_count": ",",
    "total_volume": ",.2f",
    "total_notional": ",.2f",
    "final_position": ",.4f",
    "total_fees": ",.2f",
    "final_pnl": ",.2f",
    "sharpe": ",.3f",
    "sortino": ",.3f",
    "sharpe_std": ",.3f",
    "pct_positive_buckets": ".1%",
}


def _summary_table_html(summary: dict) -> str:
    """Render the full summary as a styled table, excluding nested dicts."""
    rows = []
    for k, v in summary.items():
        if isinstance(v, (dict, list)):
            continue
        fmt = _METRIC_FORMAT.get(k)
        if fmt:
            formatted = f"{v:{fmt}}"
        elif isinstance(v, float):
            formatted = f"{v:,.4f}"
        elif isinstance(v, int):
            formatted = f"{v:,}"
        else:
            formatted = str(v)
        label = k.replace("_", " ").title()
        rows.append(f"<tr><th>{label}</th><td>{formatted}</td></tr>")
    return f'<table class="summary-table"><tbody>{"".join(rows)}</tbody></table>'


def _render_config(report: "BacktestReport") -> str | None:
    """Render the backtest config as a YAML code block."""
    config = getattr(report, "_config", None)
    if not config:
        return None
    try:
        import yaml
        config_str = yaml.dump(config, default_flow_style=False, sort_keys=False)
    except ImportError:
        import json
        config_str = json.dumps(config, indent=2, default=str)
    return f'<pre class="config-block"><code>{config_str}</code></pre>'


def _render_kpi(report: "BacktestReport") -> str:
    return _kpi_cards_html(report.summary())


def _render_summary_table(report: "BacktestReport") -> str:
    return _summary_table_html(report.summary())


def _render_pnl(report: "BacktestReport", **kwargs) -> go.Figure:
    return plot_pnl(report, **kwargs)


def _render_position(report: "BacktestReport", **kwargs) -> go.Figure:
    return plot_position(report, **kwargs)


def _render_spread(report: "BacktestReport", **kwargs) -> go.Figure:
    return plot_spread(report, **kwargs)


def _render_per_symbol(report: "BacktestReport", **kwargs) -> go.Figure | None:
    if len(report.pnl_by_symbol.columns) <= 1:
        return None
    return plot_pnl_by_symbol(report, **kwargs)


def _render_fills(report: "BacktestReport") -> str:
    return _fills_table_html(report)


def _render_warnings(report: "BacktestReport") -> str | None:
    warnings = report.summary().get("warnings") or []
    if not warnings:
        return None
    total = len(warnings)
    truncated = total > MAX_WARNINGS_DISPLAY
    display = warnings[:MAX_WARNINGS_DISPLAY]
    items = "".join(f"<li>{w}</li>" for w in display)
    note = f'<p class="muted">Showing first {MAX_WARNINGS_DISPLAY:,} of {total:,} warnings.</p>' if truncated else ""
    return f'<ol class="warnings-list">{items}</ol>{note}'


def _render_listings(report: "BacktestReport") -> str | None:
    config = getattr(report, "_config", None)
    if not config:
        return None
    raw_listings = config.get("listings")
    if not raw_listings:
        return None

    listing_ids: list[int] = []
    profile_by_id: dict[int, str] = {}
    for entry in raw_listings:
        if isinstance(entry, dict):
            lid = entry.get("listing_id")
            if lid is not None:
                listing_ids.append(int(lid))
                profile_by_id[int(lid)] = entry.get("profile", "")
        elif isinstance(entry, int):
            listing_ids.append(entry)

    if not listing_ids:
        return None

    ctx = _get_listing_context(report)

    rows = []
    for lid in listing_ids:
        listing = ctx.listings.get(lid)
        security = ctx.securities.get(listing.security_id) if listing else None
        exchange = ctx.exchanges.get(listing.exchange_id) if listing else None

        sym = security.symbol if security else "—"
        if security:
            sec_url = f"{ctx.controller_ui}/security-master/securities/{security.security_id}"
            sym_cell = f'{sym} (<a class="listing-link" href="{sec_url}">{security.security_id}</a>)'
        else:
            sym_cell = sym

        exch_name = exchange.exchange_name if exchange else "—"
        exch_cell = f"{exch_name} ({exchange.exchange_id})" if exchange else exch_name

        sec_type = SecurityType(security.type).name.capitalize() if security else "—"
        exch_sym = (listing.exchange_security_symbol or "—") if listing else "—"
        profile = profile_by_id.get(lid, "")
        url = f"{ctx.controller_ui}/security-master/listings/{lid}"
        rows.append(
            f'<tr>'
            f'<td><a class="listing-link" href="{url}">{lid}</a></td>'
            f'<td>{sym_cell}</td>'
            f'<td>{exch_cell}</td>'
            f'<td>{exch_sym}</td>'
            f'<td>{sec_type}</td>'
            f'<td>{profile}</td>'
            f'</tr>'
        )

    headers = ["Listing ID", "Symbol", "Exchange", "Exchange Symbol", "Type", "Profile"]
    header_html = "".join(f"<th>{h}</th>" for h in headers)
    rows_html = "".join(rows)
    return (
        f'<table class="listings-table">'
        f'<thead><tr>{header_html}</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table>'
    )


DEFAULT_SECTIONS: list[ReportSection] = [
    ReportSection("kpi", "Summary", _render_kpi),
    ReportSection("listings", "Listings", _render_listings),
    ReportSection("config", "Configuration", _render_config),
    ReportSection("summary_table", "Details", _render_summary_table),
    ReportSection("pnl", "Performance", _render_pnl),
    ReportSection("position", "Position & Execution", _render_position),
    ReportSection("spread", "Market Conditions", _render_spread),
    ReportSection("per_symbol", "Per-Symbol Breakdown", _render_per_symbol),
    ReportSection("fills", "Fills", _render_fills),
    ReportSection("warnings", "Warnings", _render_warnings),
]


def _render_cross_exchange(report: "BacktestReport", **kwargs) -> go.Figure:
    return plot_cross_exchange_spread(report, **kwargs)


SECTION_REGISTRY: dict[str, ReportSection] = {
    "cross_exchange_spread": ReportSection("cross_exchange_spread", "Cross-Exchange Spread", _render_cross_exchange),
}


def resolve_sections(config: dict) -> tuple[list[str], list[ReportSection], int | None, bool]:
    """Read the ``report`` key from a backtest config and resolve sections.

    Returns ``(exclude, extra_sections, max_points, plotly_cdn)``.
    """
    report_cfg = config.get("report") if config else None
    if not report_cfg:
        return [], [], None, True

    exclude = list(report_cfg.get("exclude") or [])
    max_points = report_cfg.get("max_points")
    plotly_cdn = bool(report_cfg.get("plotly_cdn", True))

    extra: list[ReportSection] = []
    for entry in report_cfg.get("sections") or []:
        name = entry if isinstance(entry, str) else entry.get("name")
        if not name:
            continue
        section = SECTION_REGISTRY.get(name)
        if not section:
            continue
        args = entry.get("args", {}) if isinstance(entry, dict) else {}
        if args:
            bound_render = functools.partial(section.render, **args)
            extra.append(ReportSection(section.name, section.title, bound_render))
        else:
            extra.append(section)

    return exclude, extra, max_points, plotly_cdn


def assemble_html(
    report: "BacktestReport",
    *,
    exclude: list[str] | None = None,
    extra_sections: list[ReportSection] | None = None,
    extra_figs: list[go.Figure] | None = None,
    max_points: int | None = DEFAULT_MAX_POINTS,
    plotly_cdn: bool = True,
) -> str:
    """Build a standalone HTML report with configurable sections."""
    from datetime import datetime as dt

    excluded = set(exclude or [])

    all_sections = list(DEFAULT_SECTIONS)
    if extra_sections:
        all_sections.extend(extra_sections)
    if extra_figs:
        for i, fig in enumerate(extra_figs):
            all_sections.append(ReportSection(
                f"_extra_{i}", f"Additional Chart {i + 1}", lambda r, f=fig: f,
            ))

    if not report._market_df.empty:
        start = report._market_df.index.min()
        end = report._market_df.index.max()
        date_range = f"{start} &mdash; {end}"
    else:
        date_range = "N/A"

    plotlyjs_included = False
    body_parts: list[str] = []

    for sec in all_sections:
        if sec.name in excluded:
            continue
        try:
            result = sec.render(report, max_points=max_points)
        except TypeError:
            result = sec.render(report)
        if result is None:
            continue

        if isinstance(result, go.Figure):
            chart_height = result.layout.height or 500
            if not plotlyjs_included:
                include_js = "cdn" if plotly_cdn else True
            else:
                include_js = False
            html = result.to_html(
                full_html=False,
                include_plotlyjs=include_js,
                config={"responsive": True},
                default_height=f"{chart_height}px",
                default_width="100%",
            )
            plotlyjs_included = True
        else:
            html = result

        body_parts.append(f'<section><h2>{sec.title}</h2>\n{html}\n</section>')

    body_html = "\n".join(body_parts)

    generated_at = dt.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        from importlib.metadata import version
        ver = version("gnomepy")
    except Exception:
        ver = "unknown"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Backtest Report</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 1200px; margin: 0 auto; padding: 24px;
    background: #f8f9fa; color: #1e293b;
  }}
  header {{
    border-bottom: 2px solid #e2e8f0; padding-bottom: 16px; margin-bottom: 24px;
  }}
  header h1 {{ margin: 0 0 4px 0; font-size: 1.5rem; color: #0f172a; }}
  header .meta {{ font-size: 0.85rem; color: #64748b; }}
  .kpi-row {{
    display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 28px;
  }}
  .kpi-card {{
    flex: 1 1 150px; background: #fff; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 16px 20px; min-width: 140px;
  }}
  .kpi-value {{
    font-size: 1.35rem; font-weight: 700; font-family: "SF Mono", "Fira Code", monospace;
  }}
  .kpi-label {{
    font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;
    letter-spacing: 0.05em; margin-top: 4px;
  }}
  section {{ margin-bottom: 36px; }}
  h2 {{
    font-size: 1.1rem; color: #334155; border-bottom: 1px solid #e2e8f0;
    padding-bottom: 8px; margin-bottom: 16px;
  }}
  .summary-table {{
    border-collapse: collapse; width: 100%; margin: 16px 0;
    font-size: 0.9rem;
  }}
  .summary-table th, .summary-table td {{
    border: 1px solid #e2e8f0; padding: 8px 14px;
  }}
  .summary-table th {{
    background: #f1f5f9; text-align: left; font-weight: 600; width: 220px;
  }}
  .summary-table td {{
    text-align: right; font-family: "SF Mono", "Fira Code", monospace; font-size: 0.85rem;
  }}
  .summary-table tr:nth-child(even) {{ background: #f8fafc; }}
  .fills-scroll {{
    max-height: 400px; overflow-y: auto; border: 1px solid #e2e8f0;
    border-radius: 8px; margin-top: 12px;
  }}
  .fills-table {{
    border-collapse: collapse; width: 100%; font-size: 0.8rem;
  }}
  .fills-table th, .fills-table td {{
    border-bottom: 1px solid #f1f5f9; padding: 6px 12px; text-align: right;
  }}
  .fills-table th {{
    background: #f1f5f9; position: sticky; top: 0; text-align: left;
    font-weight: 600; font-size: 0.75rem; text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  .fills-table td {{ font-family: "SF Mono", "Fira Code", monospace; }}
  .fills-table tbody tr:nth-child(even) {{ background: #f8fafc; }}
  .fills-table tr:hover {{ background: #eef4ff; }}
  .fills-table td span[title] {{ cursor: help; border-bottom: 1px dotted #94a3b8; }}
  .plotly-graph-div {{ width: 100% !important; min-height: 350px; }}
  .js-plotly-plot .plotly {{ min-height: 350px; }}
  .listings-table {{
    border-collapse: collapse; width: 100%; font-size: 0.9rem; margin: 4px 0;
  }}
  .listings-table th, .listings-table td {{
    border: 1px solid #e2e8f0; padding: 8px 14px;
  }}
  .listings-table th {{
    background: #f1f5f9; text-align: left; font-weight: 600;
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em;
  }}
  .listings-table td {{
    font-family: "SF Mono", "Fira Code", monospace; font-size: 0.85rem;
  }}
  .listings-table tr:nth-child(even) {{ background: #f8fafc; }}
  .listings-table tr:hover {{ background: #f0f9ff; }}
  .listing-link {{ color: #2563eb; text-decoration: none; }}
  .listing-link:hover {{ text-decoration: underline; }}
  .config-block {{
    background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 16px 20px; overflow-x: auto;
    font-family: "SF Mono", "Fira Code", monospace; font-size: 0.82rem;
    line-height: 1.5; color: #334155;
  }}
  .config-block code {{ background: none; padding: 0; }}
  .muted {{ color: #94a3b8; font-style: italic; }}
  .warnings-list {{
    max-height: 400px; overflow-y: auto; margin: 0; padding: 0 0 0 1.5em;
    border-left: 4px solid #f59e0b; background: #fffbeb;
    border-radius: 0 8px 8px 0; padding: 16px 20px 16px 2.5em;
    font-family: "SF Mono", "Fira Code", monospace; font-size: 0.82rem;
    color: #78350f; line-height: 1.6;
  }}
  .warnings-list li {{ margin-bottom: 4px; word-break: break-word; }}
  footer {{
    margin-top: 40px; padding-top: 16px; border-top: 1px solid #e2e8f0;
    font-size: 0.75rem; color: #94a3b8;
  }}
</style>
</head>
<body>
<header>
  <h1>Backtest Report</h1>
  <div class="meta">{date_range}</div>
</header>

{body_html}

<footer>
  Generated {generated_at} &middot; gnomepy {ver}
</footer>
</body>
</html>"""
