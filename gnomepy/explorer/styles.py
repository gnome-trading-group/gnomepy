"""Color palette and layout constants for the backtest explorer."""
from __future__ import annotations

BG = "#0d1117"
PANEL_BG = "#161b22"
BORDER = "#30363d"
TEXT = "#c9d1d9"
TEXT_MUTED = "#8b949e"
TEXT_HEADER = "#f0f6fc"

MID_COLOR = "#58a6ff"
BID_FILL_COLOR = "rgba(40, 167, 69, 0.25)"
ASK_FILL_COLOR = "rgba(248, 81, 73, 0.25)"
BID_LINE_COLOR = "rgba(63, 185, 80, 0.8)"
ASK_LINE_COLOR = "rgba(248, 81, 73, 0.8)"

BUY_FILL_MARKER = "#3fb950"
SELL_FILL_MARKER = "#f85149"
BUY_FILL_MARKER_B = "#aff3be"
SELL_FILL_MARKER_B = "#fdb5b0"

INTENT_BID_COLOR = "rgba(63, 185, 80, 0.6)"
INTENT_ASK_COLOR = "rgba(248, 81, 73, 0.6)"
INTENT_BID_COLOR_B = "rgba(63, 185, 80, 0.3)"
INTENT_ASK_COLOR_B = "rgba(248, 81, 73, 0.3)"

PNL_A_COLOR = "#58a6ff"
PNL_B_COLOR = "#d29922"
PNL_DELTA_FILL = "rgba(88, 166, 255, 0.15)"
POSITION_A_COLOR = "#f0883e"
POSITION_B_COLOR = "#a371f7"

CURSOR_COLOR = "rgba(255, 215, 0, 0.7)"

SIGNAL_COLORS = ["#d29922", "#a371f7", "#79c0ff", "#56d364", "#ff7b72", "#ffa657"]
DEPTH_ALPHA = [0.9, 0.6, 0.4, 0.28, 0.18, 0.12, 0.08, 0.06, 0.04, 0.03]

CHART_MARGIN = {"l": 80, "r": 15, "t": 25, "b": 0}
CHART_FONT = {"family": "JetBrains Mono, monospace", "color": TEXT, "size": 11}
PLOTLY_TEMPLATE = "plotly_dark"

CHART_LAYOUT_BASE = {
    "template": PLOTLY_TEMPLATE,
    "paper_bgcolor": PANEL_BG,
    "plot_bgcolor": PANEL_BG,
    "font": CHART_FONT,
    "margin": CHART_MARGIN,
    "showlegend": True,
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "right", "x": 1},
    "xaxis": {"showgrid": True, "gridcolor": BORDER, "zeroline": False},
    "yaxis": {"showgrid": True, "gridcolor": BORDER, "zeroline": False, "automargin": False},
    "hovermode": "x unified",
    "uirevision": "lock",
}

TABLE_STYLE = {
    "backgroundColor": PANEL_BG,
    "color": TEXT,
    "fontFamily": "JetBrains Mono, monospace",
    "fontSize": "12px",
    "border": f"1px solid {BORDER}",
}

TABLE_HEADER_STYLE = {
    "backgroundColor": BG,
    "color": TEXT_HEADER,
    "fontWeight": "bold",
    "border": f"1px solid {BORDER}",
}

TABLE_CELL_STYLE = {
    "border": f"1px solid {BORDER}",
    "padding": "4px 8px",
    "textAlign": "right",
    "whiteSpace": "nowrap",
    "overflow": "hidden",
    "textOverflow": "ellipsis",
    "maxWidth": "120px",
}

CELL_FILL_STYLE = {"backgroundColor": "rgba(63, 185, 80, 0.08)"}
CELL_INTENT_STYLE = {"backgroundColor": "rgba(88, 166, 255, 0.06)"}
CELL_ORDER_STYLE = {"backgroundColor": "rgba(248, 81, 73, 0.06)"}

BTN_STYLE = {
    "backgroundColor": "#21262d",
    "color": TEXT,
    "border": f"1px solid {BORDER}",
    "borderRadius": "4px",
    "padding": "4px 10px",
    "cursor": "pointer",
    "fontSize": "13px",
    "fontFamily": "monospace",
}

BTN_ACTIVE_STYLE = {**BTN_STYLE, "backgroundColor": "#388bfd", "color": "#ffffff"}

APP_CSS = f"""
body {{
    background-color: {BG};
    color: {TEXT};
    font-family: JetBrains Mono, monospace;
}}
.dash-table-container .cell-markdown {{
    color: {TEXT};
}}
.Select-control {{
    background-color: {PANEL_BG} !important;
    border-color: {BORDER} !important;
}}
.Select-menu-outer {{
    background-color: {PANEL_BG} !important;
    border-color: {BORDER} !important;
}}
.VirtualizedSelectOption {{
    color: {TEXT} !important;
    background-color: {PANEL_BG} !important;
}}
.VirtualizedSelectFocusedOption {{
    background-color: {BORDER} !important;
}}
.rc-slider-track {{ background-color: #388bfd; }}
.rc-slider-handle {{ border-color: #388bfd; background-color: #388bfd; }}
.rc-slider-rail {{ background-color: {BORDER}; }}
"""
