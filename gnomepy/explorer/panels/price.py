"""Price + BBO panel with fills, intents, and optional depth visualization."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go

from gnomepy.explorer.styles import (
    ASK_FILL_COLOR,
    ASK_LINE_COLOR,
    BID_FILL_COLOR,
    BID_LINE_COLOR,
    BUY_FILL_MARKER,
    BUY_FILL_MARKER_B,
    CHART_LAYOUT_BASE,
    CURSOR_COLOR,
    DEPTH_ALPHA,
    INTENT_ASK_COLOR,
    INTENT_ASK_COLOR_B,
    INTENT_BID_COLOR,
    INTENT_BID_COLOR_B,
    MID_COLOR,
    SELL_FILL_MARKER,
    SELL_FILL_MARKER_B,
)

if TYPE_CHECKING:
    from gnomepy.explorer.data import WindowedData


def build_price_figure(
    windowed_a: WindowedData,
    windowed_b: WindowedData | None = None,
    cursor_ts: pd.Timestamp | None = None,
    t_start: pd.Timestamp | None = None,
    t_end: pd.Timestamp | None = None,
    price_decimals: int = 2,
) -> go.Figure:
    fig = go.Figure()
    layout = dict(CHART_LAYOUT_BASE)
    layout["title"] = {"text": "Price & Book", "font": {"size": 12}}
    layout["yaxis"] = {**layout.get("yaxis", {}), "tickformat": f".{price_decimals}f"}

    _add_price_traces(fig, windowed_a, label="", is_comparison_b=False, price_decimals=price_decimals)
    if windowed_b is not None:
        _add_price_traces(fig, windowed_b, label=" B", is_comparison_b=True, price_decimals=price_decimals)

    if cursor_ts is not None:
        fig.add_vline(
            x=cursor_ts.isoformat(),
            line={"color": CURSOR_COLOR, "width": 1.5, "dash": "dot"},
        )

    if t_start is not None and t_end is not None:
        layout["xaxis"] = {**layout.get("xaxis", {}), "range": [t_start.isoformat(), t_end.isoformat()]}
        layout["uirevision"] = f"{t_start.value}-{t_end.value}"

    fig.update_layout(**layout)
    return fig


def build_spread_figure(
    windowed_a: WindowedData,
    windowed_b: WindowedData | None = None,
    cursor_ts: pd.Timestamp | None = None,
    t_start: pd.Timestamp | None = None,
    t_end: pd.Timestamp | None = None,
    price_decimals: int = 2,
) -> go.Figure:
    fig = go.Figure()
    layout = dict(CHART_LAYOUT_BASE)
    layout["title"] = {"text": "Spread ($)", "font": {"size": 12}}
    layout["yaxis"] = {**layout.get("yaxis", {}), "tickformat": f".{price_decimals}f"}

    for windowed, label, is_b in [(windowed_a, "", False), (windowed_b, " B", True)]:
        if windowed is None:
            continue
        mkt = windowed.market
        if mkt.empty or "bid_price_0" not in mkt.columns or "ask_price_0" not in mkt.columns:
            continue
        spread = (mkt["ask_price_0"] - mkt["bid_price_0"]).astype(float)
        color = ASK_LINE_COLOR if not is_b else BID_LINE_COLOR
        fig.add_trace(go.Scattergl(
            x=spread.index,
            y=spread,
            mode="lines",
            name=f"Spread{label}",
            line={"color": color, "width": 1},
            opacity=0.6 if is_b else 1.0,
            hovertemplate=f"Spread: %{{y:.{price_decimals}f}}<extra></extra>",
            showlegend=True,
        ))

    if cursor_ts is not None:
        fig.add_vline(
            x=cursor_ts.isoformat(),
            line={"color": CURSOR_COLOR, "width": 1.5, "dash": "dot"},
        )

    if t_start is not None and t_end is not None:
        layout["xaxis"] = {**layout.get("xaxis", {}), "range": [t_start.isoformat(), t_end.isoformat()]}
        layout["uirevision"] = f"{t_start.value}-{t_end.value}"

    fig.update_layout(**layout)
    return fig


def _add_price_traces(
    fig: go.Figure,
    windowed: WindowedData,
    label: str,
    is_comparison_b: bool,
    price_decimals: int = 2,
) -> None:
    mkt = windowed.market
    fills = windowed.fills
    intents = windowed.intents
    depth = windowed.record_depth
    is_deep = windowed.is_deep_window

    if mkt.empty:
        return

    mid_opacity = 0.6 if is_comparison_b else 1.0
    mid_dash = "dash" if is_comparison_b else "solid"
    mid_width = 1 if is_comparison_b else 1.5

    fig.add_trace(go.Scattergl(
        x=mkt.index,
        y=mkt["mid_price"],
        mode="lines",
        name=f"Mid{label}",
        line={"color": MID_COLOR, "width": mid_width, "dash": mid_dash},
        opacity=mid_opacity,
        hovertemplate=f"Mid: %{{y:.{price_decimals}f}}<extra></extra>",
        showlegend=True,
    ))

    if not is_deep or depth == 1:
        _add_bbo_band(fig, mkt, label, is_comparison_b, price_decimals)
    else:
        _add_depth_levels(fig, mkt, depth, label, is_comparison_b)

    _add_intent_traces(fig, intents, mkt, label, is_comparison_b)
    _add_fill_markers(fig, fills, label, is_comparison_b, price_decimals)


def _add_bbo_band(
    fig: go.Figure,
    mkt: pd.DataFrame,
    label: str,
    is_b: bool,
    price_decimals: int = 2,
) -> None:
    alpha_mult = 0.5 if is_b else 1.0
    bid_fill = _alpha(BID_FILL_COLOR, alpha_mult)
    ask_fill = _alpha(ASK_FILL_COLOR, alpha_mult)

    if "bid_price_0" not in mkt.columns:
        return

    fig.add_trace(go.Scattergl(
        x=mkt.index,
        y=mkt["bid_price_0"],
        mode="lines",
        name=f"Bid{label}",
        line={"color": BID_LINE_COLOR, "width": 0.5},
        hovertemplate=f"Bid: %{{y:.{price_decimals}f}}<extra></extra>",
        showlegend=True,
        opacity=0.7 if is_b else 1.0,
    ))
    fig.add_trace(go.Scattergl(
        x=mkt.index,
        y=mkt["ask_price_0"],
        mode="lines",
        name=f"Ask{label}",
        line={"color": ASK_LINE_COLOR, "width": 0.5},
        fill="tonexty",
        fillcolor=ask_fill if not is_b else bid_fill,
        hovertemplate=f"Ask: %{{y:.{price_decimals}f}}<extra></extra>",
        showlegend=True,
        opacity=0.7 if is_b else 1.0,
    ))


def _add_depth_levels(
    fig: go.Figure,
    mkt: pd.DataFrame,
    depth: int,
    label: str,
    is_b: bool,
) -> None:
    for lvl in range(depth):
        bid_col = f"bid_price_{lvl}"
        ask_col = f"ask_price_{lvl}"
        if bid_col not in mkt.columns:
            break
        alpha = DEPTH_ALPHA[lvl] * (0.5 if is_b else 1.0)
        bid_color = f"rgba(63, 185, 80, {alpha})"
        ask_color = f"rgba(248, 81, 73, {alpha})"
        show = lvl == 0 and not is_b

        fig.add_trace(go.Scattergl(
            x=mkt.index, y=mkt[bid_col],
            mode="lines", name=f"Bid L{lvl}{label}",
            line={"color": bid_color, "width": 1 if lvl == 0 else 0.5},
            showlegend=show,
        ))
        fig.add_trace(go.Scattergl(
            x=mkt.index, y=mkt[ask_col],
            mode="lines", name=f"Ask L{lvl}{label}",
            line={"color": ask_color, "width": 1 if lvl == 0 else 0.5},
            fill="tonexty" if lvl == 0 else None,
            fillcolor=f"rgba(248, 81, 73, {alpha * 0.3})" if lvl == 0 else None,
            showlegend=show,
        ))


def _add_intent_traces(
    fig: go.Figure,
    intents: pd.DataFrame,
    mkt: pd.DataFrame,
    label: str,
    is_b: bool,
) -> None:
    if intents.empty:
        return

    bid_color = INTENT_BID_COLOR_B if is_b else INTENT_BID_COLOR
    ask_color = INTENT_ASK_COLOR_B if is_b else INTENT_ASK_COLOR
    dash = "dot"

    has_bid = "bid_price" in intents.columns and (intents["bid_price"] > 0).any()
    has_ask = "ask_price" in intents.columns and (intents["ask_price"] > 0).any()
    has_take = "take_limit_price" in intents.columns and (intents["take_limit_price"] > 0).any()

    if has_bid:
        quoted_bid = intents[intents["bid_price"] > 0]["bid_price"]
        fig.add_trace(go.Scattergl(
            x=quoted_bid.index, y=quoted_bid,
            mode="lines", name=f"Quote Bid{label}",
            line={"color": bid_color, "width": 1, "dash": dash},
            showlegend=not is_b,
        ))
    if has_ask:
        quoted_ask = intents[intents["ask_price"] > 0]["ask_price"]
        fig.add_trace(go.Scattergl(
            x=quoted_ask.index, y=quoted_ask,
            mode="lines", name=f"Quote Ask{label}",
            line={"color": ask_color, "width": 1, "dash": dash},
            showlegend=not is_b,
        ))
    if has_take:
        take = intents[intents["take_limit_price"] > 0]["take_limit_price"]
        fig.add_trace(go.Scattergl(
            x=take.index, y=take,
            mode="lines+markers", name=f"Take{label}",
            line={"color": bid_color, "width": 1, "dash": "dashdot"},
            marker={"size": 4},
            showlegend=not is_b,
        ))


def _add_fill_markers(
    fig: go.Figure,
    fills: pd.DataFrame,
    label: str,
    is_b: bool,
    price_decimals: int = 2,
) -> None:
    if fills.empty:
        return

    buy_color = BUY_FILL_MARKER_B if is_b else BUY_FILL_MARKER
    sell_color = SELL_FILL_MARKER_B if is_b else SELL_FILL_MARKER
    marker_size = 7 if not is_b else 6
    buy_symbol = "triangle-up" if not is_b else "circle"
    sell_symbol = "triangle-down" if not is_b else "circle-open"

    is_buy = fills["side"].str.upper().str.contains("BID")
    buys = fills[is_buy]
    sells = fills[~is_buy]

    price_col = "fill_price" if "fill_price" in fills.columns else fills.columns[0]

    if not buys.empty:
        hover = _fill_hover(buys, price_decimals)
        fig.add_trace(go.Scattergl(
            x=buys.index, y=buys[price_col],
            mode="markers",
            name=f"Buy{label}",
            marker={"color": buy_color, "size": marker_size, "symbol": buy_symbol},
            customdata=hover,
            hovertemplate="<b>BUY%s</b><br>%%{customdata}<extra></extra>" % label,
            showlegend=True,
        ))
    if not sells.empty:
        hover = _fill_hover(sells, price_decimals)
        fig.add_trace(go.Scattergl(
            x=sells.index, y=sells[price_col],
            mode="markers",
            name=f"Sell{label}",
            marker={"color": sell_color, "size": marker_size, "symbol": sell_symbol},
            customdata=hover,
            hovertemplate="<b>SELL%s</b><br>%%{customdata}<extra></extra>" % label,
            showlegend=True,
        ))


def _fill_hover(fills: pd.DataFrame, price_decimals: int = 2) -> list[str]:
    rows = []
    fmt = f".{price_decimals}f"
    for _, r in fills.iterrows():
        parts = [f"Price: {r.get('fill_price', ''):{fmt}}"]
        if "fill_qty" in r:
            parts.append(f"Qty: {r['fill_qty']:{fmt}}")
        if "fee" in r:
            parts.append(f"Fee: {r['fee']:{fmt}}")
        if "slippage_bps" in r:
            parts.append(f"Slip: {r['slippage_bps']:.2f} bps  (${r['slippage_usd']:{fmt}})" if "slippage_usd" in r else f"Slip: {r['slippage_bps']:.2f} bps")
        rows.append("<br>".join(parts))
    return rows


def _alpha(rgba_str: str, mult: float) -> str:
    """Scale the alpha component of an rgba(...) string."""
    if "rgba" not in rgba_str:
        return rgba_str
    inner = rgba_str.strip()[5:-1]
    parts = [p.strip() for p in inner.split(",")]
    new_alpha = float(parts[3]) * mult
    return f"rgba({parts[0]}, {parts[1]}, {parts[2]}, {new_alpha:.2f})"
