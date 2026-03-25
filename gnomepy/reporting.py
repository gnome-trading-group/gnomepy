from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gnomepy.java.recorder import BacktestResults


class BacktestReport:
    """Analysis and interactive reporting for backtest results.

    Each plot method returns a plotly Figure for notebook display.
    generate_html() combines all plots into a self-contained HTML file.

    Usage:
        results = backtest.run()
        report = BacktestReport(results)
        report.plot_pnl().show()
        report.generate_html("/tmp/report.html")
    """

    def __init__(self, results: BacktestResults):
        self.results = results
        self._cached_pnl_df = None
        self._cached_fills_df = None

    # --- Data computation ---

    def compute_pnl(self) -> pd.DataFrame:
        """Compute mark-to-market PnL by merging market and execution data.

        Returns a DataFrame indexed by market timestamp with columns:
        - price: mid price (forward-filled)
        - quantity: net position (cumulative signed fills)
        - fee: cumulative fees
        - pnl: cumulative mark-to-market PnL minus fees
        - nmv: net market value (quantity * price)
        """
        if self._cached_pnl_df is not None:
            return self._cached_pnl_df

        market_df = self.results.market_records_df()
        exec_df = self.results.execution_records_df()

        if market_df.empty:
            return pd.DataFrame()

        df = market_df[["mid_price"]].copy()
        df = df.rename(columns={"mid_price": "price"})
        df["price"] = df["price"].replace(0, np.nan).ffill().bfill()

        df["quantity"] = 0.0
        df["fee"] = 0.0

        if not exec_df.empty:
            fills = exec_df[exec_df["exec_type"].isin(["FILL", "PARTIAL_FILL"])].copy()
            if not fills.empty:
                fills["signed_qty"] = np.where(
                    fills["side"] == "Bid", fills["filled_qty"], -fills["filled_qty"]
                )

                fills_reset = fills[["signed_qty", "fee"]].reset_index()
                fills_reset["timestamp_event"] = fills_reset["timestamp_event"].astype(np.int64)
                market_reset = df[[]].reset_index()
                market_reset["timestamp"] = market_reset["timestamp"].astype(np.int64)

                merged = pd.merge_asof(
                    fills_reset.sort_values("timestamp_event"),
                    market_reset.sort_values("timestamp"),
                    left_on="timestamp_event",
                    right_on="timestamp",
                    direction="nearest",
                )

                merged["timestamp"] = pd.to_datetime(merged["timestamp"])
                grouped = merged.groupby("timestamp").agg(
                    signed_qty=("signed_qty", "sum"),
                    fee=("fee", "sum"),
                ).reindex(df.index, fill_value=0.0)

                df["quantity"] = grouped["signed_qty"].cumsum()
                df["fee"] = grouped["fee"].cumsum()

        prev_qty = df["quantity"].shift(1, fill_value=0.0)
        price_change = df["price"].diff().fillna(0.0)
        df["pnl"] = (prev_qty * price_change).cumsum() - df["fee"]
        df["nmv"] = df["quantity"] * df["price"]

        self._cached_pnl_df = df
        return df

    def _get_fills(self) -> pd.DataFrame:
        """Get fills (FILL + PARTIAL_FILL) from execution records."""
        if self._cached_fills_df is not None:
            return self._cached_fills_df

        exec_df = self.results.execution_records_df()
        if exec_df.empty:
            self._cached_fills_df = pd.DataFrame()
            return self._cached_fills_df

        fills = exec_df[exec_df["exec_type"].isin(["FILL", "PARTIAL_FILL"])].copy()
        self._cached_fills_df = fills
        return fills

    def summary(self) -> dict:
        """Minimal summary statistics for the backtest run."""
        pnl_df = self.compute_pnl()
        fills = self._get_fills()

        return {
            "total_pnl": float(pnl_df["pnl"].iloc[-1]) if not pnl_df.empty else 0.0,
            "total_fees": float(pnl_df["fee"].iloc[-1]) if not pnl_df.empty else 0.0,
            "num_fills": len(fills),
            "final_quantity": float(pnl_df["quantity"].iloc[-1]) if not pnl_df.empty else 0.0,
        }

    # --- Individual plot methods ---

    def plot_price(self) -> go.Figure:
        """Mid price with fill markers."""
        pnl_df = self.compute_pnl()
        fills = self._get_fills()

        fig = go.Figure()

        # Mid price line
        fig.add_trace(go.Scatter(
            x=pnl_df.index, y=pnl_df["price"],
            mode="lines", name="Mid Price",
            line=dict(color="#636EFA", width=1),
        ))

        # Buy fills
        if not fills.empty:
            buys = fills[fills["side"] == "Bid"]
            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys.index, y=buys["fill_price"],
                    mode="markers", name="Buy Fill",
                    marker=dict(
                        symbol="triangle-up", color="#00CC96",
                        size=buys["filled_qty"] / buys["filled_qty"].max() * 12 + 4,
                        line=dict(width=1, color="darkgreen"),
                    ),
                ))

            # Sell fills
            sells = fills[fills["side"] == "Ask"]
            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells.index, y=sells["fill_price"],
                    mode="markers", name="Sell Fill",
                    marker=dict(
                        symbol="triangle-down", color="#EF553B",
                        size=sells["filled_qty"] / sells["filled_qty"].max() * 12 + 4,
                        line=dict(width=1, color="darkred"),
                    ),
                ))

        fig.update_layout(title="Price & Fills", yaxis_title="Price", xaxis_title="Time")
        return fig

    def plot_pnl(self) -> go.Figure:
        """Cumulative PnL: net and gross with fee shading."""
        pnl_df = self.compute_pnl()

        fig = go.Figure()

        gross_pnl = pnl_df["pnl"] + pnl_df["fee"]

        # Gross PnL (filled to net = fee region)
        fig.add_trace(go.Scatter(
            x=pnl_df.index, y=gross_pnl,
            mode="lines", name="PnL (gross)",
            line=dict(color="#636EFA", width=1),
        ))

        # Net PnL
        fig.add_trace(go.Scatter(
            x=pnl_df.index, y=pnl_df["pnl"],
            mode="lines", name="PnL (net of fees)",
            line=dict(color="#00CC96", width=1.5),
            fill="tonexty", fillcolor="rgba(239, 85, 59, 0.15)",
        ))

        fig.update_layout(title="Cumulative PnL", yaxis_title="PnL", xaxis_title="Time")
        return fig

    def plot_position(self) -> go.Figure:
        """Position over time with long/short coloring."""
        pnl_df = self.compute_pnl()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=pnl_df.index, y=pnl_df["quantity"],
            mode="lines", name="Position",
            line=dict(color="#AB63FA", width=1, shape="hv"),
            fill="tozeroy",
            fillcolor="rgba(171, 99, 250, 0.15)",
        ))

        fig.update_layout(title="Position", yaxis_title="Quantity", xaxis_title="Time")
        return fig

    def plot_drawdown(self) -> go.Figure:
        """Drawdown from peak PnL."""
        pnl_df = self.compute_pnl()

        cumulative_pnl = pnl_df["pnl"]
        peak = cumulative_pnl.cummax()
        drawdown = cumulative_pnl - peak

        max_dd = drawdown.min()
        max_dd_time = drawdown.idxmin()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=pnl_df.index, y=drawdown,
            mode="lines", name="Drawdown",
            line=dict(color="#EF553B", width=1),
            fill="tozeroy", fillcolor="rgba(239, 85, 59, 0.2)",
        ))

        fig.add_annotation(
            x=max_dd_time, y=max_dd,
            text=f"Max DD: {max_dd:.2f}",
            showarrow=True, arrowhead=2,
            font=dict(color="#EF553B"),
        )

        fig.update_layout(title="Drawdown", yaxis_title="Drawdown", xaxis_title="Time")
        return fig

    def plot_spread(self) -> go.Figure:
        """Spread over time."""
        market_df = self.results.market_records_df()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=market_df.index, y=market_df["spread"],
            mode="lines", name="Spread",
            line=dict(color="#FFA15A", width=1),
        ))

        fig.update_layout(title="Spread", yaxis_title="Spread", xaxis_title="Time")
        return fig

    def plot_fees(self) -> go.Figure:
        """Cumulative fees over time."""
        pnl_df = self.compute_pnl()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=pnl_df.index, y=pnl_df["fee"],
            mode="lines", name="Cumulative Fees",
            line=dict(color="#EF553B", width=1.5),
            fill="tozeroy", fillcolor="rgba(239, 85, 59, 0.1)",
        ))

        fig.update_layout(title="Cumulative Fees", yaxis_title="Fees", xaxis_title="Time")
        return fig

    def plot_fills(self) -> go.Figure:
        """Fill scatter with mid price background."""
        pnl_df = self.compute_pnl()
        fills = self._get_fills()

        fig = go.Figure()

        # Mid price background
        fig.add_trace(go.Scatter(
            x=pnl_df.index, y=pnl_df["price"],
            mode="lines", name="Mid Price",
            line=dict(color="#CCCCCC", width=1),
        ))

        if not fills.empty:
            buys = fills[fills["side"] == "Bid"]
            sells = fills[fills["side"] == "Ask"]

            max_qty = fills["filled_qty"].max() if not fills.empty else 1

            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys.index, y=buys["fill_price"],
                    mode="markers", name="Buy",
                    marker=dict(
                        color="#00CC96",
                        size=buys["filled_qty"] / max_qty * 15 + 5,
                        line=dict(width=1, color="darkgreen"),
                    ),
                    text=[f"qty: {q:.4f}<br>price: {p:.2f}" for q, p in zip(buys["filled_qty"], buys["fill_price"])],
                    hoverinfo="text+name",
                ))

            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells.index, y=sells["fill_price"],
                    mode="markers", name="Sell",
                    marker=dict(
                        color="#EF553B",
                        size=sells["filled_qty"] / max_qty * 15 + 5,
                        line=dict(width=1, color="darkred"),
                    ),
                    text=[f"qty: {q:.4f}<br>price: {p:.2f}" for q, p in zip(sells["filled_qty"], sells["fill_price"])],
                    hoverinfo="text+name",
                ))

        fig.update_layout(title="Fill Scatter", yaxis_title="Price", xaxis_title="Time")
        return fig

    # --- HTML report ---

    def generate_html(self, path: str | Path) -> None:
        """Generate a self-contained HTML report with all plots and summary.

        Args:
            path: Output file path (e.g. "/tmp/report.html")
        """
        path = Path(path)
        summary = self.summary()

        # Build summary table HTML
        summary_html = "<div style='font-family: monospace; margin: 20px;'>"
        summary_html += "<h1>Backtest Report</h1>"
        summary_html += "<table style='border-collapse: collapse; margin-bottom: 20px;'>"
        for key, value in summary.items():
            formatted = f"{value:,.4f}" if isinstance(value, float) else str(value)
            summary_html += (
                f"<tr>"
                f"<td style='padding: 4px 12px; border: 1px solid #ddd; font-weight: bold;'>{key}</td>"
                f"<td style='padding: 4px 12px; border: 1px solid #ddd;'>{formatted}</td>"
                f"</tr>"
            )
        summary_html += "</table></div>"

        # Generate all plots
        plots = [
            self.plot_price(),
            self.plot_pnl(),
            self.plot_position(),
            self.plot_drawdown(),
            self.plot_spread(),
            self.plot_fees(),
            self.plot_fills(),
        ]

        # Build each plot as a self-contained div with plotly.js included only once
        # First plot includes plotly.js, rest don't
        plot_htmls = []
        for i, fig in enumerate(plots):
            fig.update_layout(height=400, margin=dict(l=60, r=30, t=40, b=40))
            plot_htmls.append(
                fig.to_html(
                    full_html=False,
                    include_plotlyjs=(i == 0),  # only include JS in first plot
                    div_id=f"plot-{i}",
                )
            )

        plot_divs = "\n<hr>\n".join(plot_htmls)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Backtest Report</title>
</head>
<body style="background: #fafafa; max-width: 1200px; margin: 0 auto;">
{summary_html}
{plot_divs}
</body>
</html>"""

        path.write_text(html)
