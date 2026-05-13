"""Custom strategy signal panel."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go

from gnomepy.explorer.data import MAX_CHART_POINTS, _lttb_df
from gnomepy.explorer.styles import (
    BORDER,
    CHART_FONT,
    CHART_LAYOUT_BASE,
    CHART_MARGIN,
    CURSOR_COLOR,
    PANEL_BG,
    PLOTLY_TEMPLATE,
    SIGNAL_COLORS,
)

if TYPE_CHECKING:
    from gnomepy.explorer.data import ExplorerDataStore, WindowedData


def get_signal_options(custom_dfs: dict[str, pd.DataFrame]) -> list[dict]:
    """Build Dash dropdown options from custom metric DataFrames."""
    options = []
    for buf_name, df in custom_dfs.items():
        if df.empty:
            continue
        for col in df.select_dtypes("number").columns:
            label = f"{buf_name}.{col}" if len(custom_dfs) > 1 else col
            options.append({"label": label, "value": f"{buf_name}::{col}"})
    return options


def build_signals_figure(
    windowed_a: WindowedData,
    windowed_b: WindowedData | None,
    selected_signals: list[str],
    cursor_ts: pd.Timestamp | None = None,
    t_start: pd.Timestamp | None = None,
    t_end: pd.Timestamp | None = None,
) -> go.Figure:
    fig = go.Figure()

    for i, sig_key in enumerate(selected_signals):
        color = SIGNAL_COLORS[i % len(SIGNAL_COLORS)]
        _add_signal_trace(fig, windowed_a.custom, sig_key, color, label="", dash="solid")
        if windowed_b is not None:
            _add_signal_trace(fig, windowed_b.custom, sig_key, color, label=" B", dash="dash")

    if cursor_ts is not None:
        fig.add_vline(
            x=cursor_ts.isoformat(),
            line={"color": CURSOR_COLOR, "width": 1.5, "dash": "dot"},
        )

    layout = dict(CHART_LAYOUT_BASE)
    layout["title"] = {"text": "Signals", "font": {"size": 12}}
    if t_start is not None and t_end is not None:
        layout["xaxis"] = {**layout.get("xaxis", {}), "range": [t_start.isoformat(), t_end.isoformat()]}

    fig.update_layout(**layout)
    return fig


def _add_signal_trace(
    fig: go.Figure,
    custom: dict[str, pd.DataFrame],
    sig_key: str,
    color: str,
    label: str,
    dash: str,
) -> None:
    if "::" not in sig_key:
        return
    buf_name, col = sig_key.split("::", 1)
    df = custom.get(buf_name)
    if df is None or df.empty or col not in df.columns:
        return

    series = df[col].dropna()
    if series.empty:
        return

    if len(series) > MAX_CHART_POINTS:
        series = _lttb_df(series.to_frame(col), MAX_CHART_POINTS)[col]

    display_name = f"{buf_name}.{col}{label}" if len(custom) > 1 else f"{col}{label}"
    fig.add_trace(go.Scattergl(
        x=series.index, y=series,
        mode="lines",
        name=display_name,
        line={"color": color, "width": 1.5, "dash": dash},
        opacity=0.7 if label else 1.0,
        showlegend=True,
    ))
