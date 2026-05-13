"""PnL + position panel with optional comparison overlay."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gnomepy.explorer.data import MAX_CHART_POINTS, _lttb_df, _slice
from gnomepy.explorer.styles import (
    BORDER,
    CHART_FONT,
    CHART_MARGIN,
    CURSOR_COLOR,
    PANEL_BG,
    PLOTLY_TEMPLATE,
    PNL_A_COLOR,
    PNL_B_COLOR,
    PNL_DELTA_FILL,
    POSITION_A_COLOR,
    POSITION_B_COLOR,
)

FEES_LINE_COLOR = "rgba(210, 153, 34, 0.45)"
FEES_FILL_COLOR = "rgba(210, 153, 34, 0.08)"

if TYPE_CHECKING:
    from gnomepy.explorer.data import ExplorerDataStore


def build_pnl_figure(
    store_a: ExplorerDataStore,
    store_b: ExplorerDataStore | None,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
    cursor_ts: pd.Timestamp | None = None,
    pnl_delta: pd.Series | None = None,
    price_decimals: int = 2,
) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.06,
        subplot_titles=("PnL", "Position"),
    )

    _add_pnl_traces(fig, store_a, t_start, t_end, label="A" if store_b else "", color=PNL_A_COLOR, row=1)
    _add_fees_trace(fig, store_a, t_start, t_end, price_decimals)
    _add_position_traces(fig, store_a, t_start, t_end, label="A" if store_b else "", color=POSITION_A_COLOR, row=2)

    if store_b is not None:
        _add_pnl_traces(fig, store_b, t_start, t_end, label="B", color=PNL_B_COLOR, row=1, dash="dash")
        _add_position_traces(fig, store_b, t_start, t_end, label="B", color=POSITION_B_COLOR, row=2, dash="dash")

        if pnl_delta is not None and not pnl_delta.empty:
            delta_win = pnl_delta.loc[t_start:t_end]
            if not delta_win.empty:
                fig.add_trace(go.Scattergl(
                    x=delta_win.index, y=delta_win,
                    mode="lines", name="ΔPnL (A-B)",
                    line={"color": PNL_DELTA_FILL.replace("0.15", "0.6"), "width": 1},
                    fill="tozeroy", fillcolor=PNL_DELTA_FILL,
                    showlegend=True,
                ), row=1, col=1)

    _add_zero_line(fig, row=1)
    _add_zero_line(fig, row=2)

    x_range = [t_start.isoformat(), t_end.isoformat()]
    if cursor_ts is not None:
        for row in (1, 2):
            fig.add_vline(
                x=cursor_ts.isoformat(),
                line={"color": CURSOR_COLOR, "width": 1.5, "dash": "dot"},
                row=row, col=1,
            )

    tick_fmt = f".{price_decimals}f"
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=PANEL_BG,
        font=CHART_FONT,
        margin=CHART_MARGIN,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "right", "x": 1},
        hovermode="x unified",
        uirevision=f"{t_start.value}-{t_end.value}",
        yaxis3={
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "showticklabels": False,
            "ticks": "",
            "zeroline": False,
            "automargin": False,
            "fixedrange": True,
        },
    )
    fig.update_xaxes(showgrid=True, gridcolor=BORDER, zeroline=False, range=x_range)
    fig.update_yaxes(showgrid=True, gridcolor=BORDER, zeroline=False, automargin=False)
    fig.update_yaxes(tickprefix="$", tickformat=tick_fmt, row=1, col=1)
    fig.update_yaxes(tickformat=tick_fmt, row=2, col=1)
    return fig


def _add_pnl_traces(
    fig: go.Figure,
    store: ExplorerDataStore,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
    label: str,
    color: str,
    row: int,
    dash: str = "solid",
) -> None:
    pnl = _slice(store.curves.pnl, t_start, t_end)
    if pnl.empty:
        return
    if len(pnl) > MAX_CHART_POINTS:
        pnl = _lttb_df(pnl.to_frame("pnl"), MAX_CHART_POINTS)["pnl"]

    name = f"PnL {label}" if label else "PnL"
    fig.add_trace(go.Scattergl(
        x=pnl.index, y=pnl,
        mode="lines", name=name,
        line={"color": color, "width": 1.5, "dash": dash},
        showlegend=True,
    ), row=row, col=1)


def _add_fees_trace(
    fig: go.Figure,
    store: ExplorerDataStore,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
    price_decimals: int = 2,
) -> None:
    fees = _slice(store.curves.fees, t_start, t_end)
    if fees.empty:
        return
    if len(fees) > MAX_CHART_POINTS:
        fees = _lttb_df(fees.to_frame("fees"), MAX_CHART_POINTS)["fees"]

    fmt = f".{price_decimals}f"
    fig.add_trace(go.Scattergl(
        x=fees.index, y=fees,
        xaxis="x", yaxis="y3",
        mode="lines", name="Fees",
        line={"color": FEES_LINE_COLOR, "width": 1},
        fill="tozeroy", fillcolor=FEES_FILL_COLOR,
        hovertemplate=f"Fees: $%{{y:{fmt}}}<extra></extra>",
        showlegend=True,
    ))


def _add_position_traces(
    fig: go.Figure,
    store: ExplorerDataStore,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
    label: str,
    color: str,
    row: int,
    dash: str = "solid",
) -> None:
    pos = _slice(store.curves.position, t_start, t_end)
    if pos.empty:
        return
    if len(pos) > MAX_CHART_POINTS:
        pos = _lttb_df(pos.to_frame("position"), MAX_CHART_POINTS)["position"]

    name = f"Pos {label}" if label else "Position"
    fig.add_trace(go.Scattergl(
        x=pos.index, y=pos,
        mode="lines", name=name,
        line={"color": color, "width": 1.5, "dash": dash, "shape": "hv"},
        showlegend=True,
    ), row=row, col=1)


def _add_zero_line(fig: go.Figure, row: int) -> None:
    fig.add_hline(
        y=0,
        line={"color": "rgba(255,255,255,0.15)", "width": 1, "dash": "dot"},
        row=row, col=1,
    )
