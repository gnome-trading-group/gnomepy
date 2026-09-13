"""Book Analysis tab: DOM ladder, context chart, and fill replay."""
from __future__ import annotations

from typing import TYPE_CHECKING

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

from gnomepy.explorer.styles import (
    ASK_FILL_COLOR,
    ASK_LINE_COLOR,
    BID_LINE_COLOR,
    BORDER,
    CHART_LAYOUT_BASE,
    CURSOR_COLOR,
    FILL_FLASH_COLOR,
    INTENT_LADDER_BORDER,
    LADDER_ASK_BAR,
    LADDER_BG,
    LADDER_BID_BAR,
    LADDER_HEADER_BG,
    LADDER_MID_HIGHLIGHT,
    LADDER_PRICE_TEXT,
    MID_COLOR,
    PANEL_BG,
    TEXT_MUTED,
)

if TYPE_CHECKING:
    from gnomepy.explorer.data import ExplorerDataStore, WindowedData

_PRICE_TOL = 1e-7


def _safe_pct(size: float, max_size: float) -> int:
    if max_size <= 0:
        return 0
    return min(100, max(0, int(size / max_size * 100)))


def _bid_bar_style(size: float, max_size: float) -> dict:
    pct = _safe_pct(size, max_size)
    return {"background": f"linear-gradient(to right, {LADDER_BID_BAR} {pct}%, transparent {pct}%)"}


def _ask_bar_style(size: float, max_size: float) -> dict:
    pct = _safe_pct(size, max_size)
    return {"background": f"linear-gradient(to left, {LADDER_ASK_BAR} {pct}%, transparent {pct}%)"}


def _near_fill(price: float, recent_fills: pd.DataFrame) -> tuple[bool, float]:
    """Return (is_near, fill_qty) for the nearest fill at this price level."""
    if recent_fills.empty or "fill_price" not in recent_fills.columns:
        return False, 0.0
    mask = recent_fills["fill_price"].astype(float).sub(price).abs() < _PRICE_TOL
    if not mask.any():
        return False, 0.0
    return True, float(recent_fills[mask].iloc[0]["fill_qty"])


def _merge_with_intent(
    levels: list[tuple[float, float]],
    intent_price: float,
    intent_size: float,
) -> list[tuple[float, float, bool]]:
    """Returns (price, size, is_intent) rows ordered high-to-low price."""
    if intent_price <= 0:
        return [(p, s, False) for p, s in levels]

    for p, _ in levels:
        if abs(p - intent_price) < _PRICE_TOL:
            return [(p, s, abs(p - intent_price) < _PRICE_TOL) for p, s in levels]

    result: list[tuple[float, float, bool]] = [(p, s, False) for p, s in levels]
    for i, (p, _, _) in enumerate(result):
        if intent_price > p:
            result.insert(i, (intent_price, intent_size, True))
            return result
    result.append((intent_price, intent_size, True))
    return result


def _price_cell(price: float, price_decimals: int, is_intent: bool, is_bid_intent: bool) -> html.Td:
    style: dict = {
        "textAlign": "center",
        "color": INTENT_LADDER_BORDER if is_intent else LADDER_PRICE_TEXT,
        "padding": "2px 6px",
        "fontFamily": "monospace",
        "fontSize": "12px",
        "minWidth": "70px",
    }
    if is_intent:
        side = "borderLeft" if is_bid_intent else "borderRight"
        style[side] = f"2px dashed {INTENT_LADDER_BORDER}"
    return html.Td(f"{price:.{price_decimals}f}", style=style)


def build_ladder(
    book_data: dict | None,
    intent_data: dict | None,
    recent_fills: pd.DataFrame,
    venue_label: str,
    price_decimals: int,
    record_depth: int,
) -> html.Div:
    if not book_data:
        return html.Div(
            f"No book data for {venue_label}",
            style={"color": TEXT_MUTED, "fontFamily": "monospace", "fontSize": "12px", "padding": "16px", "textAlign": "center"},
        )

    bids: list[tuple[float, float]] = book_data.get("bids", [])
    asks: list[tuple[float, float]] = book_data.get("asks", [])
    ts: pd.Timestamp | None = book_data.get("timestamp")

    intent = intent_data or {}
    intent_bid_price = float(intent.get("bid_price", 0) or 0)
    intent_bid_size = float(intent.get("bid_size", 0) or 0)
    intent_ask_price = float(intent.get("ask_price", 0) or 0)
    intent_ask_size = float(intent.get("ask_size", 0) or 0)

    all_sizes = [s for _, s in bids] + [s for _, s in asks]
    max_size = max(all_sizes) if all_sizes else 1.0

    # asks displayed high-to-low (deepest at top, best ask at bottom before spread)
    ask_rows = _merge_with_intent(list(reversed(asks)), intent_ask_price, intent_ask_size)
    # bids already high-to-low (best bid at top)
    bid_rows = _merge_with_intent(bids, intent_bid_price, intent_bid_size)

    ts_str = ts.strftime("%H:%M:%S.%f")[:-3] if ts is not None else "—"

    table_rows: list[html.Tr] = [
        html.Tr([
            html.Th("Ask Size", style={"textAlign": "right", "color": TEXT_MUTED, "fontSize": "11px", "padding": "3px 6px", "fontWeight": "normal", "borderBottom": f"1px solid {BORDER}"}),
            html.Th("Price", style={"textAlign": "center", "color": TEXT_MUTED, "fontSize": "11px", "padding": "3px 6px", "fontWeight": "normal", "borderBottom": f"1px solid {BORDER}", "minWidth": "70px"}),
            html.Th("Bid Size", style={"textAlign": "right", "color": TEXT_MUTED, "fontSize": "11px", "padding": "3px 6px", "fontWeight": "normal", "borderBottom": f"1px solid {BORDER}"}),
        ])
    ]

    for ap, as_, is_intent in ask_rows:
        is_fill, fill_qty = _near_fill(ap, recent_fills)
        bar = _ask_bar_style(as_, max_size)
        size_label = f"{as_:.4g}"
        if is_fill:
            size_label += f" ▶{fill_qty:.4g}"
        size_style: dict = {
            "textAlign": "right",
            "color": "#f85149",
            "padding": "2px 6px",
            "fontFamily": "monospace",
            "fontSize": "12px",
            **bar,
        }
        if is_intent:
            size_style["borderLeft"] = f"2px dashed {INTENT_LADDER_BORDER}"
            size_style["color"] = INTENT_LADDER_BORDER
        table_rows.append(html.Tr([
            html.Td(size_label, style=size_style),
            _price_cell(ap, price_decimals, is_intent, False),
            html.Td("", style={"padding": "2px 6px"}),
        ], className="fill-flash" if is_fill else ""))

    best_bid = bids[0][0] if bids else 0.0
    best_ask = asks[0][0] if asks else 0.0
    spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0.0
    spread_label = f"── spread {spread:.{price_decimals}f} ──"
    table_rows.append(html.Tr([
        html.Td(spread_label, colSpan=3, style={
            "textAlign": "center",
            "color": TEXT_MUTED,
            "fontSize": "11px",
            "padding": "4px 6px",
            "backgroundColor": LADDER_MID_HIGHLIGHT,
            "borderTop": f"1px solid {BORDER}",
            "borderBottom": f"1px solid {BORDER}",
        }),
    ]))

    for bp, bs, is_intent in bid_rows:
        is_fill, fill_qty = _near_fill(bp, recent_fills)
        bar = _bid_bar_style(bs, max_size)
        size_label = f"{bs:.4g}"
        if is_fill:
            size_label += f" ◀{fill_qty:.4g}"
        size_style = {
            "textAlign": "right",
            "color": "#3fb950",
            "padding": "2px 6px",
            "fontFamily": "monospace",
            "fontSize": "12px",
            **bar,
        }
        if is_intent:
            size_style["borderLeft"] = f"2px dashed {INTENT_LADDER_BORDER}"
            size_style["color"] = INTENT_LADDER_BORDER
        table_rows.append(html.Tr([
            html.Td("", style={"padding": "2px 6px"}),
            _price_cell(bp, price_decimals, is_intent, True),
            html.Td(size_label, style=size_style),
        ], className="fill-flash" if is_fill else ""))

    header = html.Div([
        html.Span(venue_label, style={"color": LADDER_PRICE_TEXT, "fontFamily": "monospace", "fontSize": "12px", "fontWeight": "bold"}),
        html.Span(f"  {ts_str}", style={"color": TEXT_MUTED, "fontFamily": "monospace", "fontSize": "11px", "marginLeft": "8px"}),
    ], style={"backgroundColor": LADDER_HEADER_BG, "padding": "5px 8px", "borderBottom": f"1px solid {BORDER}", "borderRadius": "4px 4px 0 0"})

    table = html.Table(
        table_rows,
        className="ladder-table",
        style={"width": "100%"},
    )

    return html.Div([
        header,
        html.Div(table, style={"overflowY": "auto", "maxHeight": "55vh"}),
    ], style={"backgroundColor": LADDER_BG, "border": f"1px solid {BORDER}", "borderRadius": "4px"})


def build_book_tab_layout(
    listings: list[tuple[int, int]],
    store: ExplorerDataStore,
) -> html.Div:
    n = len(listings)
    placeholder_cols: list[dbc.Col] = []
    if n == 0:
        placeholder_cols = [dbc.Col(
            html.Div("No listings found", style={"color": TEXT_MUTED, "textAlign": "center", "padding": "20px", "fontFamily": "monospace", "fontSize": "12px"}),
            width=6,
        )]
    else:
        col_width = max(3, 12 // n)
        for eid, sid in listings:
            label = store.listing_label(eid, sid)
            placeholder_cols.append(dbc.Col(
                html.Div(
                    f"Navigate to a timestamp to view {label}",
                    style={"color": TEXT_MUTED, "textAlign": "center", "padding": "20px", "fontFamily": "monospace", "fontSize": "12px"},
                ),
                width=col_width,
            ))

    return html.Div([
        dbc.Row(dbc.Col(
            dcc.Loading(
                dcc.Graph(
                    id="book-context-chart",
                    config={"displayModeBar": False, "scrollZoom": True, "doubleClick": "reset+autosize", "responsive": True},
                    style={"height": "15vh"},
                ),
                type="circle",
                color="#58a6ff",
                style={"height": "15vh"},
            ),
            width=12,
        ), className="g-0 mt-1"),
        html.Div(
            id="book-ladders-container",
            children=[dbc.Row(placeholder_cols, className="g-2 mt-1")],
        ),
    ])


def build_book_context_chart(
    windowed: WindowedData,
    cursor_ts: pd.Timestamp | None,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
    price_decimals: int,
) -> go.Figure:
    fig = go.Figure()
    layout = dict(CHART_LAYOUT_BASE)
    layout["title"] = {"text": "Price Context", "font": {"size": 11}}
    layout["yaxis"] = {**layout.get("yaxis", {}), "tickformat": f".{price_decimals}f"}
    layout["xaxis"] = {**layout.get("xaxis", {}), "range": [t_start.isoformat(), t_end.isoformat()]}
    layout["margin"] = {"l": 60, "r": 10, "t": 22, "b": 0}
    layout["showlegend"] = False

    mkt = windowed.market
    if not mkt.empty:
        if "mid_price" in mkt.columns:
            fig.add_trace(go.Scattergl(
                x=mkt.index, y=mkt["mid_price"],
                mode="lines", name="Mid",
                line={"color": MID_COLOR, "width": 1},
            ))
        if "bid_price_0" in mkt.columns:
            fig.add_trace(go.Scattergl(
                x=mkt.index, y=mkt["bid_price_0"],
                mode="lines", name="Bid",
                line={"color": BID_LINE_COLOR, "width": 0.5},
            ))
            fig.add_trace(go.Scattergl(
                x=mkt.index, y=mkt["ask_price_0"],
                mode="lines", name="Ask",
                line={"color": ASK_LINE_COLOR, "width": 0.5},
                fill="tonexty",
                fillcolor=ASK_FILL_COLOR,
            ))

    if cursor_ts is not None:
        fig.add_vline(
            x=cursor_ts.isoformat(),
            line={"color": CURSOR_COLOR, "width": 2, "dash": "dot"},
        )

    fig.update_layout(**layout)
    return fig
