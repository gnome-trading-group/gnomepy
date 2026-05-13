"""Plotly Dash application for the backtest explorer.

Module-level globals hold the loaded ExplorerDataStore instances.  This is
the standard Dash pattern for large datasets that should not be serialized
to the browser via dcc.Store.
"""
from __future__ import annotations

import json

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dash_table, dcc, html, no_update

from gnomepy.explorer.data import (
    SLIDER_RESOLUTION,
    ComparisonStore,
    ExplorerDataStore,
)
from gnomepy.explorer.panels.event_log import build_event_columns, build_event_records
from gnomepy.explorer.panels.price import build_price_figure, build_spread_figure
from gnomepy.explorer.panels.pnl import build_pnl_figure
from gnomepy.explorer.panels.signals import build_signals_figure, get_signal_options
from gnomepy.explorer.styles import (
    APP_CSS,
    BG,
    BORDER,
    BTN_ACTIVE_STYLE,
    BTN_STYLE,
    PANEL_BG,
    TABLE_CELL_STYLE,
    TABLE_HEADER_STYLE,
    TABLE_STYLE,
    TEXT,
    TEXT_MUTED,
)

_STORE_A: ExplorerDataStore | None = None
_STORE_B: ExplorerDataStore | None = None
_COMPARISON: ComparisonStore | None = None
_PRICE_DECIMALS: int = 2

_SPEED_INTERVALS = {1: 600, 5: 120, 10: 60}

_CHART_CONFIG = {
    "displayModeBar": False,
    "scrollZoom": True,
    "doubleClick": "reset+autosize",
    "responsive": True,
}


def create_app(
    store_a: ExplorerDataStore,
    store_b: ExplorerDataStore | None = None,
    price_decimals: int = 2,
) -> dash.Dash:
    global _STORE_A, _STORE_B, _COMPARISON, _PRICE_DECIMALS
    _STORE_A = store_a
    _STORE_B = store_b
    _COMPARISON = ComparisonStore(store_a, store_b) if store_b else None
    _PRICE_DECIMALS = price_decimals

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        title="Gnome Backtest Explorer",
        suppress_callback_exceptions=True,
    )
    app.index_string = app.index_string.replace(
        "</head>",
        f"<style>{APP_CSS}</style></head>",
    )

    t_min = store_a.t_min
    t_max = store_a.t_max
    initial_window = {"t_start": t_min.isoformat(), "t_end": t_max.isoformat()}

    signal_options = get_signal_options(store_a.custom_dfs)
    default_signals = [o["value"] for o in signal_options[:3]]
    has_signals = bool(signal_options)

    slider_marks = _build_slider_marks(t_min, t_max)

    app.layout = dbc.Container(
        [
            dcc.Store(id="time-window", data=initial_window),
            dcc.Store(id="cursor-pos", data=None),
            dcc.Store(id="play-state", data={"playing": False, "speed": 1}),
            dcc.Interval(id="play-interval", interval=600, disabled=True, n_intervals=0),

            _build_header(store_a, store_b),
            _build_config_diff_section(),
            _build_nav_bar(slider_marks, signal_options, default_signals, has_signals, price_decimals),
            _build_window_label(),

            dbc.Row(dbc.Col(
                dcc.Graph(id="price-chart", config=_CHART_CONFIG, style={"height": "35vh"}),
                width=12,
            ), className="g-0 mt-1"),

            dbc.Row(dbc.Col(
                dcc.Graph(id="spread-chart", config=_CHART_CONFIG, style={"height": "12vh"}),
                width=12,
            ), className="g-0"),

            dbc.Row(dbc.Col(
                dcc.Graph(id="signals-chart", config=_CHART_CONFIG, style={"height": "20vh"}),
                width=12,
            ), id="signals-row", className="g-0",
                style={"display": "block" if has_signals else "none"}),

            dbc.Row(dbc.Col(
                dcc.Graph(id="pnl-chart", config=_CHART_CONFIG, style={"height": "25vh"}),
                width=12,
            ), className="g-0"),

            dbc.Row(dbc.Col(
                _build_event_table(),
                width=12,
            ), className="g-0 mt-1 mb-3"),
        ],
        fluid=True,
        style={"backgroundColor": BG, "minHeight": "100vh", "padding": "8px"},
    )

    _register_callbacks(app, t_min, t_max, has_signals, default_signals, price_decimals)
    return app


def _build_header(store_a: ExplorerDataStore, store_b: ExplorerDataStore | None) -> dbc.Row:
    meta_a = store_a.metadata
    backtest_id = meta_a.backtest_id if meta_a else "—"
    strategy = (meta_a.strategy or "").split(":")[-1] if meta_a else "—"
    dates = ""
    if meta_a and meta_a.start_date and meta_a.end_date:
        dates = f"{meta_a.start_date} → {meta_a.end_date}"

    right_items = []
    if store_b is not None:
        meta_b = store_b.metadata
        bid_b = meta_b.backtest_id if meta_b else "B"
        right_items.append(
            dbc.Badge(f"Comparing: {bid_b[:12]}…", color="warning", className="ms-2")
        )

    return dbc.Row([
        dbc.Col([
            html.H6(backtest_id, style={"color": TEXT, "fontFamily": "monospace", "margin": 0}),
            html.Small(f"{strategy}  {dates}", style={"color": TEXT_MUTED, "fontFamily": "monospace"}),
        ], width="auto"),
        dbc.Col(right_items, width="auto", className="ms-auto d-flex align-items-center"),
    ], className="px-2 py-2", style={"borderBottom": f"1px solid {BORDER}", "backgroundColor": PANEL_BG})


def _build_config_diff_section() -> html.Div:
    return html.Div(id="config-diff-section")


def _build_nav_bar(
    slider_marks: dict,
    signal_options: list[dict],
    default_signals: list[str],
    has_signals: bool,
    price_decimals: int = 2,
) -> dbc.Row:
    step_buttons = dbc.ButtonGroup([
        dbc.Button("|◀", id="btn-first", n_clicks=0, size="sm", outline=True, color="secondary"),
        dbc.Button("◀", id="btn-prev", n_clicks=0, size="sm", outline=True, color="secondary"),
        dbc.Button("▶", id="btn-play", n_clicks=0, size="sm", color="primary"),
        dbc.Button("▶", id="btn-next", n_clicks=0, size="sm", outline=True, color="secondary"),
        dbc.Button("▶|", id="btn-last", n_clicks=0, size="sm", outline=True, color="secondary"),
    ], size="sm")

    event_filter = dcc.Dropdown(
        id="event-filter",
        options=[
            {"label": "All events", "value": "all"},
            {"label": "Fills", "value": "fill"},
            {"label": "Intents", "value": "intent"},
            {"label": "Orders", "value": "order"},
        ],
        value="fill",
        clearable=False,
        style={"width": "140px", "fontSize": "12px"},
    )

    speed_select = dcc.Dropdown(
        id="speed-select",
        options=[
            {"label": "1×", "value": 1},
            {"label": "5×", "value": 5},
            {"label": "10×", "value": 10},
        ],
        value=1,
        clearable=False,
        style={"width": "70px", "fontSize": "12px"},
    )

    reset_btn = dbc.Button(
        "Reset", id="btn-reset", n_clicks=0, size="sm", outline=True, color="light"
    )

    signals_row = []
    if has_signals:
        signals_row = [
            html.Span("Signals:", style={"color": TEXT_MUTED, "fontSize": "12px", "marginRight": "6px"}),
            dcc.Dropdown(
                id="signal-selector",
                options=signal_options,
                value=default_signals,
                multi=True,
                clearable=True,
                style={"minWidth": "240px", "fontSize": "12px"},
            ),
        ]
    else:
        signals_row = [html.Div(id="signal-selector", style={"display": "none"})]

    controls_row = dbc.Row([
        dbc.Col(step_buttons, width="auto"),
        dbc.Col(html.Div([
            html.Span("Event:", style={"color": TEXT_MUTED, "fontSize": "12px", "marginRight": "4px"}),
            event_filter,
        ], style={"display": "flex", "alignItems": "center", "gap": "4px"}), width="auto"),
        dbc.Col(html.Div([
            html.Span("Speed:", style={"color": TEXT_MUTED, "fontSize": "12px", "marginRight": "4px"}),
            speed_select,
        ], style={"display": "flex", "alignItems": "center", "gap": "4px"}), width="auto"),
        dbc.Col(html.Div(signals_row, style={"display": "flex", "alignItems": "center", "gap": "4px"}), width="auto"),
        dbc.Col(html.Div([
            html.Span("Decimals:", style={"color": TEXT_MUTED, "fontSize": "12px", "marginRight": "4px"}),
            dcc.Input(
                id="price-decimals-input",
                type="number", min=0, max=12, step=1,
                value=price_decimals,
                debounce=True,
                style={"width": "52px", "fontSize": "12px", "backgroundColor": "#161b22", "color": TEXT, "border": f"1px solid {BORDER}", "borderRadius": "4px", "padding": "2px 4px"},
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": "4px"}), width="auto"),
        dbc.Col(reset_btn, width="auto", className="ms-auto"),
    ], align="center", className="g-2")

    slider = dcc.RangeSlider(
        id="time-slider",
        min=0, max=SLIDER_RESOLUTION,
        step=1,
        value=[0, SLIDER_RESOLUTION],
        marks=slider_marks,
        tooltip={"placement": "bottom", "always_visible": False},
        allowCross=False,
    )

    return dbc.Row(dbc.Col([
        controls_row,
        html.Div(slider, style={"marginTop": "8px", "marginBottom": "4px"}),
    ], width=12), style={"backgroundColor": PANEL_BG, "padding": "8px", "borderBottom": f"1px solid {BORDER}"})


def _build_window_label() -> html.Div:
    return html.Div(
        id="window-label",
        style={"color": TEXT_MUTED, "fontFamily": "monospace", "fontSize": "11px", "padding": "2px 8px"},
    )


def _build_event_table() -> dash_table.DataTable:
    return dash_table.DataTable(
        id="event-table",
        columns=build_event_columns(_PRICE_DECIMALS),
        data=[],
        page_size=100,
        page_action="native",
        sort_action="native",
        filter_action="native",
        row_selectable="single",
        selected_rows=[],
        fixed_rows={"headers": True},
        style_table={"overflowX": "auto", "height": "22vh", "overflowY": "auto"},
        style_header=TABLE_HEADER_STYLE,
        style_cell=TABLE_CELL_STYLE,
        style_data=TABLE_STYLE,
        style_data_conditional=[
            {"if": {"filter_query": '{type} = "fill"'}, "backgroundColor": "rgba(63, 185, 80, 0.07)"},
            {"if": {"filter_query": '{type} = "intent"'}, "backgroundColor": "rgba(88, 166, 255, 0.05)"},
            {"if": {"filter_query": '{type} = "order"'}, "backgroundColor": "rgba(248, 81, 73, 0.05)"},
            {"if": {"column_id": "source", "filter_query": '{source} = "B"'}, "color": "#d29922"},
        ],
        style_cell_conditional=[
            {"if": {"column_id": "time"}, "textAlign": "left", "minWidth": "110px"},
            {"if": {"column_id": "type"}, "textAlign": "left", "minWidth": "55px"},
            {"if": {"column_id": "order_type"}, "textAlign": "left", "minWidth": "65px"},
            {"if": {"column_id": "oid"}, "textAlign": "right", "minWidth": "50px"},
            {"if": {"column_id": "source"}, "textAlign": "center", "minWidth": "40px"},
            {"if": {"column_id": "side"}, "textAlign": "left", "minWidth": "60px"},
            {"if": {"column_id": "status"}, "textAlign": "left", "minWidth": "80px"},
        ],
    )


def _build_slider_marks(t_min: pd.Timestamp, t_max: pd.Timestamp) -> dict:
    span = (t_max - t_min).total_seconds()
    marks = {
        0: {"label": t_min.strftime("%m/%d %H:%M"), "style": {"color": TEXT_MUTED, "fontSize": "10px"}},
        SLIDER_RESOLUTION: {"label": t_max.strftime("%m/%d %H:%M"), "style": {"color": TEXT_MUTED, "fontSize": "10px"}},
    }
    for frac in (0.25, 0.5, 0.75):
        ts = t_min + pd.Timedelta(seconds=span * frac)
        val = int(SLIDER_RESOLUTION * frac)
        marks[val] = {"label": ts.strftime("%m/%d %H:%M"), "style": {"color": TEXT_MUTED, "fontSize": "10px"}}
    return marks


def _parse_relayout(
    relayout_data: dict | None,
    t_min: pd.Timestamp,
    t_max: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if not relayout_data:
        return None
    if relayout_data.get("xaxis.autorange") or relayout_data.get("autosize"):
        return t_min, t_max
    t0 = relayout_data.get("xaxis.range[0]")
    t1 = relayout_data.get("xaxis.range[1]")
    if t0 is not None and t1 is not None:
        try:
            return pd.Timestamp(t0), pd.Timestamp(t1)
        except Exception:
            return None
    r = relayout_data.get("xaxis.range")
    if isinstance(r, list) and len(r) == 2:
        try:
            return pd.Timestamp(r[0]), pd.Timestamp(r[1])
        except Exception:
            return None
    return None


def _window_from_store(data: dict | None, t_min: pd.Timestamp, t_max: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not data:
        return t_min, t_max
    t_start = pd.Timestamp(data.get("t_start", t_min.isoformat()))
    t_end = pd.Timestamp(data.get("t_end", t_max.isoformat()))
    return (
        max(t_min, min(t_max, t_start)),
        max(t_min, min(t_max, t_end)),
    )


def _clamp_window(
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
    t_min: pd.Timestamp,
    t_max: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    t_start = max(t_min, t_start)
    t_end = min(t_max, t_end)
    if t_end <= t_start:
        t_end = t_start + pd.Timedelta(seconds=1)
    return t_start, t_end


def _step_event(
    event_ts_array: np.ndarray,
    cursor_ns: int | None,
    direction: int,
) -> int | None:
    if len(event_ts_array) == 0:
        return None
    if cursor_ns is None:
        return int(event_ts_array[0] if direction > 0 else event_ts_array[-1])
    idx = int(np.searchsorted(event_ts_array, cursor_ns, side="right" if direction > 0 else "left"))
    if direction > 0:
        if idx >= len(event_ts_array):
            return None
        return int(event_ts_array[idx])
    else:
        idx -= 1
        if idx < 0:
            return None
        return int(event_ts_array[idx])


def _center_window(
    cursor_ns: int,
    current_start: pd.Timestamp,
    current_end: pd.Timestamp,
    t_min: pd.Timestamp,
    t_max: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    width = current_end - current_start
    cursor_ts = pd.Timestamp(cursor_ns)
    new_start = cursor_ts - width / 2
    new_end = cursor_ts + width / 2
    return _clamp_window(new_start, new_end, t_min, t_max)


def _register_callbacks(
    app: dash.Dash,
    t_min: pd.Timestamp,
    t_max: pd.Timestamp,
    has_signals: bool,
    default_signals: list[str],
    price_decimals: int = 2,
) -> None:

    @app.callback(
        Output("time-window", "data"),
        Output("cursor-pos", "data"),
        [
            Input("price-chart", "relayoutData"),
            Input("spread-chart", "relayoutData"),
            Input("pnl-chart", "relayoutData"),
            Input("signals-chart", "relayoutData"),
            Input("time-slider", "value"),
            Input("btn-first", "n_clicks"),
            Input("btn-prev", "n_clicks"),
            Input("btn-next", "n_clicks"),
            Input("btn-last", "n_clicks"),
            Input("btn-reset", "n_clicks"),
            Input("play-interval", "n_intervals"),
            Input("event-table", "selected_rows"),
        ],
        [
            State("time-window", "data"),
            State("cursor-pos", "data"),
            State("event-filter", "value"),
            State("play-state", "data"),
            State("event-table", "data"),
        ],
        prevent_initial_call=True,
    )
    def master_time_control(
        price_relayout, spread_relayout, pnl_relayout, signals_relayout,
        slider_value,
        first_clicks, prev_clicks, next_clicks, last_clicks, reset_clicks,
        play_intervals,
        selected_rows,
        current_window, cursor_pos, event_filter, play_state, table_data,
    ):
        triggered_id = ctx.triggered_id
        t_start, t_end = _window_from_store(current_window, t_min, t_max)

        if triggered_id in ("price-chart", "spread-chart", "pnl-chart", "signals-chart"):
            relayout = {
                "price-chart": price_relayout,
                "spread-chart": spread_relayout,
                "pnl-chart": pnl_relayout,
                "signals-chart": signals_relayout,
            }[triggered_id]
            result = _parse_relayout(relayout, t_min, t_max)
            if result is None:
                return no_update, no_update
            new_start, new_end = _clamp_window(result[0], result[1], t_min, t_max)
            new_window = {"t_start": new_start.isoformat(), "t_end": new_end.isoformat()}
            return new_window, cursor_pos

        if triggered_id == "time-slider" and slider_value is not None:
            sl_start = _STORE_A.slider_to_ts(slider_value[0])
            sl_end = _STORE_A.slider_to_ts(slider_value[1])
            new_start, new_end = _clamp_window(sl_start, sl_end, t_min, t_max)
            return {"t_start": new_start.isoformat(), "t_end": new_end.isoformat()}, cursor_pos

        if triggered_id == "btn-reset":
            return {"t_start": t_min.isoformat(), "t_end": t_max.isoformat()}, None

        if triggered_id in ("btn-first", "btn-prev", "btn-next", "btn-last", "play-interval"):
            if triggered_id == "play-interval" and not (play_state or {}).get("playing", False):
                return no_update, no_update

            filter_val = event_filter if event_filter != "all" else None
            event_types = [filter_val] if filter_val else None
            ev_ts = _STORE_A.event_timestamps(event_types)

            if triggered_id in ("btn-first",):
                new_cursor = int(ev_ts[0]) if len(ev_ts) > 0 else None
            elif triggered_id in ("btn-last",):
                new_cursor = int(ev_ts[-1]) if len(ev_ts) > 0 else None
            elif triggered_id in ("btn-prev",):
                new_cursor = _step_event(ev_ts, cursor_pos, direction=-1)
            else:
                new_cursor = _step_event(ev_ts, cursor_pos, direction=1)

            if new_cursor is None:
                return no_update, no_update

            new_start, new_end = _center_window(new_cursor, t_start, t_end, t_min, t_max)
            return {"t_start": new_start.isoformat(), "t_end": new_end.isoformat()}, new_cursor

        if triggered_id == "event-table" and selected_rows and table_data:
            row_idx = selected_rows[0]
            if row_idx < len(table_data):
                row = table_data[row_idx]
                iso = row.get("timestamp_iso")
                if iso:
                    event_ts = pd.Timestamp(iso)
                    new_cursor = event_ts.value
                    new_start, new_end = _center_window(new_cursor, t_start, t_end, t_min, t_max)
                    return {"t_start": new_start.isoformat(), "t_end": new_end.isoformat()}, new_cursor

        return no_update, no_update

    @app.callback(
        Output("play-state", "data"),
        Output("play-interval", "disabled"),
        Output("play-interval", "interval"),
        Output("btn-play", "children"),
        Input("btn-play", "n_clicks"),
        Input("speed-select", "value"),
        State("play-state", "data"),
        prevent_initial_call=True,
    )
    def toggle_play(n_clicks, speed, play_state):
        triggered_id = ctx.triggered_id
        playing = (play_state or {}).get("playing", False)

        if triggered_id == "speed-select":
            interval_ms = _SPEED_INTERVALS.get(speed or 1, 600)
            return {**play_state, "speed": speed}, not playing, interval_ms, "⏸" if playing else "▶"

        playing = not playing
        interval_ms = _SPEED_INTERVALS.get((play_state or {}).get("speed", 1), 600)
        return {"playing": playing, "speed": (play_state or {}).get("speed", 1)}, not playing, interval_ms, "⏸" if playing else "▶"

    @app.callback(
        Output("price-chart", "figure"),
        Input("time-window", "data"),
        Input("cursor-pos", "data"),
        Input("price-decimals-input", "value"),
        prevent_initial_call=False,
    )
    def render_price(window_data, cursor_pos, decimals):
        t_start, t_end = _window_from_store(window_data, t_min, t_max)
        cursor_ts = pd.Timestamp(cursor_pos) if cursor_pos else None
        pd_ = int(decimals) if decimals is not None else price_decimals

        windowed_a = _STORE_A.window(t_start, t_end)
        windowed_b = _STORE_B.window(t_start, t_end) if _STORE_B else None

        return build_price_figure(windowed_a, windowed_b, cursor_ts, t_start, t_end, pd_)

    @app.callback(
        Output("spread-chart", "figure"),
        Input("time-window", "data"),
        Input("cursor-pos", "data"),
        Input("price-decimals-input", "value"),
        prevent_initial_call=False,
    )
    def render_spread(window_data, cursor_pos, decimals):
        t_start, t_end = _window_from_store(window_data, t_min, t_max)
        cursor_ts = pd.Timestamp(cursor_pos) if cursor_pos else None
        pd_ = int(decimals) if decimals is not None else price_decimals

        windowed_a = _STORE_A.window(t_start, t_end)
        windowed_b = _STORE_B.window(t_start, t_end) if _STORE_B else None

        return build_spread_figure(windowed_a, windowed_b, cursor_ts, t_start, t_end, pd_)

    @app.callback(
        Output("event-table", "columns"),
        Input("price-decimals-input", "value"),
        prevent_initial_call=False,
    )
    def update_table_columns(decimals):
        pd_ = int(decimals) if decimals is not None else price_decimals
        return build_event_columns(pd_)

    @app.callback(
        Output("pnl-chart", "figure"),
        Input("time-window", "data"),
        Input("cursor-pos", "data"),
        Input("price-decimals-input", "value"),
        prevent_initial_call=False,
    )
    def render_pnl(window_data, cursor_pos, decimals):
        t_start, t_end = _window_from_store(window_data, t_min, t_max)
        cursor_ts = pd.Timestamp(cursor_pos) if cursor_pos else None
        pnl_delta = _COMPARISON.pnl_delta(t_start, t_end) if _COMPARISON else None
        pd_ = int(decimals) if decimals is not None else price_decimals

        return build_pnl_figure(_STORE_A, _STORE_B, t_start, t_end, cursor_ts, pnl_delta, pd_)

    if has_signals:
        @app.callback(
            Output("signals-chart", "figure"),
            Input("time-window", "data"),
            Input("cursor-pos", "data"),
            Input("signal-selector", "value"),
            prevent_initial_call=False,
        )
        def render_signals(window_data, cursor_pos, selected_signals):
            t_start, t_end = _window_from_store(window_data, t_min, t_max)
            cursor_ts = pd.Timestamp(cursor_pos) if cursor_pos else None
            sigs = selected_signals or default_signals

            windowed_a = _STORE_A.window(t_start, t_end)
            windowed_b = _STORE_B.window(t_start, t_end) if _STORE_B else None

            return build_signals_figure(windowed_a, windowed_b, sigs, cursor_ts, t_start, t_end)

    @app.callback(
        Output("event-table", "data"),
        Input("time-window", "data"),
        prevent_initial_call=False,
    )
    def render_event_table(window_data):
        t_start, t_end = _window_from_store(window_data, t_min, t_max)
        windowed_a = _STORE_A.window(t_start, t_end)
        windowed_b = _STORE_B.window(t_start, t_end) if _STORE_B else None
        return build_event_records(windowed_a, windowed_b)

    @app.callback(
        Output("window-label", "children"),
        Input("time-window", "data"),
        prevent_initial_call=False,
    )
    def render_window_label(window_data):
        t_start, t_end = _window_from_store(window_data, t_min, t_max)
        span = t_end - t_start
        total_s = span.total_seconds()
        if total_s >= 86400:
            span_str = f"{total_s / 86400:.1f}d"
        elif total_s >= 3600:
            span_str = f"{total_s / 3600:.1f}h"
        elif total_s >= 60:
            span_str = f"{total_s / 60:.1f}m"
        else:
            span_str = f"{total_s:.1f}s"
        return f"Window: {t_start.strftime('%Y-%m-%d %H:%M:%S')} — {t_end.strftime('%H:%M:%S')}  ({span_str})"

    @app.callback(
        Output("config-diff-section", "children"),
        Input("time-window", "data"),
        prevent_initial_call=False,
    )
    def render_config_diff(_):
        if _COMPARISON is None:
            return None
        diffs = _COMPARISON.config_diff()
        if not diffs:
            return None
        rows = []
        for k, (va, vb) in diffs.items():
            rows.append(html.Tr([
                html.Td(k, style={"fontFamily": "monospace", "fontSize": "12px", "padding": "2px 8px"}),
                html.Td(str(va), style={"color": "#58a6ff", "fontFamily": "monospace", "fontSize": "12px", "padding": "2px 8px"}),
                html.Td(str(vb), style={"color": "#d29922", "fontFamily": "monospace", "fontSize": "12px", "padding": "2px 8px"}),
            ]))
        table = html.Table([
            html.Thead(html.Tr([
                html.Th("Param", style={"padding": "2px 8px", "fontSize": "11px"}),
                html.Th("A", style={"padding": "2px 8px", "fontSize": "11px", "color": "#58a6ff"}),
                html.Th("B", style={"padding": "2px 8px", "fontSize": "11px", "color": "#d29922"}),
            ])),
            html.Tbody(rows),
        ], style={"borderCollapse": "collapse", "width": "auto"})

        return dbc.Accordion([
            dbc.AccordionItem(table, title="Config Differences"),
        ], start_collapsed=True, flush=True,
            style={"backgroundColor": PANEL_BG, "borderBottom": f"1px solid {BORDER}"})
