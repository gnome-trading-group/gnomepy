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
        - spread_capture: cumulative PnL from filling at favorable prices vs mid
        - holding_pnl: cumulative PnL from price movement on existing position
        - pnl: total PnL (holding + spread_capture - fees)
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

        # Initialize per-tick changes to zero
        qty_changes = pd.Series(0.0, index=df.index)
        fee_changes = pd.Series(0.0, index=df.index)
        spread_changes = pd.Series(0.0, index=df.index)

        if not exec_df.empty:
            fills = exec_df[
                exec_df["exec_type"].isin(["FILL", "PARTIAL_FILL"])
                & (exec_df["fill_price"] > 0)
                & (exec_df["filled_qty"] > 0)
                & (exec_df["side"].isin(["Bid", "Ask"]))
            ].copy()
            if not fills.empty:
                fills["signed_qty"] = np.where(
                    fills["side"] == "Bid", fills["filled_qty"], -fills["filled_qty"]
                )

                market_times = df.index.asi8
                prices = df["price"].values
                for _, fill in fills.iterrows():
                    fill_ns = fill.name.value
                    # Forward: first market tick at or after the fill
                    pos = np.searchsorted(market_times, fill_ns, side="left")
                    if pos >= len(market_times):
                        pos = len(market_times) - 1
                    qty_changes.iloc[pos] += fill["signed_qty"]
                    fee_changes.iloc[pos] += fill["fee"]
                    # Spread capture: signed_qty * (mid - fill_price)
                    # Buy below mid = positive, sell above mid = positive
                    spread_changes.iloc[pos] += fill["signed_qty"] * (prices[pos] - fill["fill_price"])

        df["quantity"] = qty_changes.cumsum()
        df["fee"] = fee_changes.cumsum()
        df["spread_capture"] = spread_changes.cumsum()

        prev_qty = df["quantity"].shift(1, fill_value=0.0)
        price_change = df["price"].diff().fillna(0.0)
        df["holding_pnl"] = (prev_qty * price_change).cumsum()
        df["pnl"] = df["holding_pnl"] + df["spread_capture"] - df["fee"]
        df["nmv"] = df["quantity"] * df["price"]

        self._cached_pnl_df = df
        return df

    def _get_fills(self) -> pd.DataFrame:
        """Get valid fills (FILL + PARTIAL_FILL) from execution records.

        Filters out records with invalid data (zero/negative prices, missing side).
        """
        if self._cached_fills_df is not None:
            return self._cached_fills_df

        exec_df = self.results.execution_records_df()
        if exec_df.empty:
            self._cached_fills_df = pd.DataFrame()
            return self._cached_fills_df

        fills = exec_df[
            exec_df["exec_type"].isin(["FILL", "PARTIAL_FILL"])
            & (exec_df["fill_price"] > 0)
            & (exec_df["filled_qty"] > 0)
            & (exec_df["side"].isin(["Bid", "Ask"]))
        ].copy()
        self._cached_fills_df = fills
        return fills

    def summary(self) -> dict:
        """Summary statistics including microstructure quality metrics."""
        pnl_df = self.compute_pnl()
        fills = self._get_fills()
        market_df = self.results.market_records_df()

        result = {
            "total_pnl": float(pnl_df["pnl"].iloc[-1]) if not pnl_df.empty else 0.0,
            "total_fees": float(pnl_df["fee"].iloc[-1]) if not pnl_df.empty else 0.0,
            "spread_capture": float(pnl_df["spread_capture"].iloc[-1]) if not pnl_df.empty else 0.0,
            "holding_pnl": float(pnl_df["holding_pnl"].iloc[-1]) if not pnl_df.empty else 0.0,
            "num_fills": len(fills),
            "final_quantity": float(pnl_df["quantity"].iloc[-1]) if not pnl_df.empty else 0.0,
        }

        if not fills.empty and not pnl_df.empty:
            # Duration
            duration_sec = (fills.index[-1] - fills.index[0]).total_seconds()
            duration_min = max(duration_sec / 60, 1e-9)

            # 1. Fill rate — fills per intent
            intent_count = self.results.intent_record_count
            result["fills_per_minute"] = round(len(fills) / duration_min, 1)
            result["fill_rate_pct"] = round(len(fills) / max(1, intent_count) * 100, 2)

            # 2. Fill size distribution
            result["avg_fill_size"] = round(float(fills["filled_qty"].mean()), 6)
            result["median_fill_size"] = round(float(fills["filled_qty"].median()), 6)

            # 3. Adverse selection — % of fills immediately underwater
            market_times = pnl_df.index.asi8
            prices = pnl_df["price"].values
            underwater = 0
            for _, fill in fills.iterrows():
                fill_ns = fill.name.value
                pos = np.searchsorted(market_times, fill_ns)
                if pos >= len(market_times):
                    pos = len(market_times) - 1
                elif pos > 0 and abs(fill_ns - market_times[pos - 1]) < abs(fill_ns - market_times[pos]):
                    pos = pos - 1
                mid = prices[pos]
                if fill["side"] == "Bid" and fill["fill_price"] > mid:
                    underwater += 1
                elif fill["side"] == "Ask" and fill["fill_price"] < mid:
                    underwater += 1
            result["adverse_selection_pct"] = round(underwater / len(fills) * 100, 1)

            # 4. Spread capture per fill vs market spread
            avg_spread = float(market_df["spread"].mean()) if not market_df.empty else 0.0
            sc_per_fill = float(pnl_df["spread_capture"].iloc[-1]) / len(fills) if len(fills) > 0 else 0.0
            result["avg_spread_capture_per_fill"] = round(sc_per_fill, 4)
            result["avg_market_spread"] = round(avg_spread, 4)
            result["capture_vs_spread_pct"] = round(sc_per_fill / avg_spread * 100, 1) if avg_spread > 0 else 0.0

        return result

    def _add_price_overlay(self, fig: go.Figure) -> None:
        """Add a grey price line on a secondary y-axis."""
        pnl_df = self.compute_pnl()
        fig.add_trace(go.Scatter(
            x=pnl_df.index, y=pnl_df["price"],
            mode="lines", name="Price",
            line=dict(color="rgba(180, 180, 180, 0.4)", width=1),
            yaxis="y2",
        ))
        fig.update_layout(
            yaxis2=dict(
                overlaying="y", side="right",
                showgrid=False,
                title=dict(text="Price", font=dict(color="rgba(180, 180, 180, 0.6)")),
                tickfont=dict(color="rgba(180, 180, 180, 0.6)"),
            ),
        )

    # --- Individual plot methods ---

    def plot_price(self, show_fills: bool = False) -> go.Figure:
        """Mid price with optional fill markers."""
        pnl_df = self.compute_pnl()

        fig = go.Figure()

        # Mid price line
        fig.add_trace(go.Scatter(
            x=pnl_df.index, y=pnl_df["price"],
            mode="lines", name="Mid Price",
            line=dict(color="#636EFA", width=1),
        ))

        if show_fills:
            fills = self._get_fills()
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

        self._add_price_overlay(fig)
        fig.update_layout(title="Cumulative PnL", yaxis_title="PnL", xaxis_title="Time")
        return fig

    def plot_pnl_decomposition(self) -> go.Figure:
        """PnL decomposition: spread capture vs holding PnL vs fees."""
        pnl_df = self.compute_pnl()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=pnl_df.index, y=pnl_df["holding_pnl"],
            mode="lines", name="Holding PnL",
            line=dict(color="#636EFA", width=1),
        ))

        fig.add_trace(go.Scatter(
            x=pnl_df.index, y=pnl_df["spread_capture"],
            mode="lines", name="Spread Capture",
            line=dict(color="#00CC96", width=1),
        ))

        fig.add_trace(go.Scatter(
            x=pnl_df.index, y=-pnl_df["fee"],
            mode="lines", name="Fees (negative)",
            line=dict(color="#EF553B", width=1),
        ))

        fig.add_trace(go.Scatter(
            x=pnl_df.index, y=pnl_df["pnl"],
            mode="lines", name="Total PnL",
            line=dict(color="white", width=2, dash="dot"),
        ))

        self._add_price_overlay(fig)
        fig.update_layout(
            title="PnL Decomposition",
            yaxis_title="PnL",
            xaxis_title="Time",
        )
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

        self._add_price_overlay(fig)
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

        self._add_price_overlay(fig)
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

    def plot_quote_vs_spread(self) -> go.Figure:
        """Quoted spread vs market spread over time."""
        market_df = self.results.market_records_df()
        intent_df = self.results.intent_records_df()

        if market_df.empty or intent_df.empty:
            return go.Figure()

        fig = go.Figure()

        # Market spread
        fig.add_trace(go.Scatter(
            x=market_df.index, y=market_df["spread"],
            mode="lines", name="Market Spread",
            line=dict(color="#636EFA", width=1),
        ))

        # Quoted spread from intents
        quoted_spread = intent_df["ask_price"] - intent_df["bid_price"]
        valid = (intent_df["bid_size"] > 0) & (intent_df["ask_size"] > 0)
        quoted_spread = quoted_spread[valid]

        if not quoted_spread.empty:
            fig.add_trace(go.Scatter(
                x=quoted_spread.index, y=quoted_spread,
                mode="lines", name="Quoted Spread",
                line=dict(color="#00CC96", width=1),
            ))

            # Ratio line on secondary axis
            # Align market spread to intent timestamps using merge_asof
            qs_reset = quoted_spread.reset_index()
            qs_reset.columns = ["timestamp", "quoted_spread"]
            qs_reset["timestamp"] = qs_reset["timestamp"].astype(np.int64)
            ms_reset = market_df[["spread"]].reset_index()
            ms_reset.columns = ["timestamp", "market_spread"]
            ms_reset["timestamp"] = ms_reset["timestamp"].astype(np.int64)
            merged = pd.merge_asof(
                qs_reset.sort_values("timestamp"),
                ms_reset.sort_values("timestamp"),
                on="timestamp", direction="nearest")
            merged["timestamp"] = pd.to_datetime(merged["timestamp"])
            ratio = merged.set_index("timestamp")
            ratio["ratio"] = ratio["quoted_spread"] / ratio["market_spread"].replace(0, np.nan)
            fig.add_trace(go.Scatter(
                x=ratio.index, y=ratio["ratio"],
                mode="lines", name="Quote/Market Ratio",
                line=dict(color="#FFA15A", width=1, dash="dot"),
                yaxis="y2",
            ))
            fig.update_layout(
                yaxis2=dict(
                    overlaying="y", side="right",
                    showgrid=False,
                    title=dict(text="Ratio", font=dict(color="#FFA15A")),
                    tickfont=dict(color="#FFA15A"),
                ),
            )

        fig.update_layout(title="Quoted vs Market Spread", yaxis_title="Spread", xaxis_title="Time")
        return fig

    def quote_quality(self) -> pd.DataFrame:
        """Statistics on quote placement relative to the market.

        Returns a DataFrame with metrics for bid and ask sides:
        - distance_mean: avg distance from best bid/ask (positive = behind, negative = inside)
        - distance_median: median distance
        - pct_at_best: % of time quoting at the best level
        - pct_inside: % of time quoting inside the spread
        - pct_behind: % of time quoting behind the best level
        - quoted_spread_mean: avg quoted spread
        - market_spread_mean: avg market spread
        - spread_ratio_mean: avg quoted/market spread ratio
        """
        market_df = self.results.market_records_df()
        intent_df = self.results.intent_records_df()

        if market_df.empty or intent_df.empty:
            return pd.DataFrame()

        # Align intents with market data
        intent_reset = intent_df[["bid_price", "bid_size", "ask_price", "ask_size"]].reset_index()
        intent_reset["timestamp"] = intent_reset["timestamp"].astype(np.int64)
        market_reset = market_df[["best_bid_price", "best_ask_price", "spread"]].reset_index()
        market_reset["timestamp"] = market_reset["timestamp"].astype(np.int64)
        aligned = pd.merge_asof(
            intent_reset.sort_values("timestamp"),
            market_reset.sort_values("timestamp"),
            on="timestamp", direction="nearest")

        bid_valid = aligned["bid_size"] > 0
        ask_valid = aligned["ask_size"] > 0

        bid_dist = (aligned["best_bid_price"] - aligned["bid_price"])[bid_valid]
        ask_dist = (aligned["ask_price"] - aligned["best_ask_price"])[ask_valid]

        quoted_spread = (aligned["ask_price"] - aligned["bid_price"])[bid_valid & ask_valid]
        market_spread = aligned["spread"][bid_valid & ask_valid]

        def side_stats(dist, label):
            if dist.empty:
                return {}
            return {
                "side": label,
                "distance_mean": round(float(dist.mean()), 4),
                "distance_median": round(float(dist.median()), 4),
                "pct_at_best": round(float((dist == 0).mean() * 100), 1),
                "pct_inside": round(float((dist < 0).mean() * 100), 1),
                "pct_behind": round(float((dist > 0).mean() * 100), 1),
            }

        rows = []
        rows.append(side_stats(bid_dist, "Bid"))
        rows.append(side_stats(ask_dist, "Ask"))

        if not quoted_spread.empty and not market_spread.empty:
            ratio = quoted_spread / market_spread.replace(0, np.nan)
            rows.append({
                "side": "Spread",
                "distance_mean": round(float(quoted_spread.mean()), 4),
                "distance_median": round(float(quoted_spread.median()), 4),
                "pct_at_best": round(float(ratio.mean()), 2),
                "pct_inside": round(float(market_spread.mean()), 4),
                "pct_behind": None,
            })
            # Rename columns for spread row
            rows[-1] = {
                "side": "Spread",
                "quoted_spread_mean": round(float(quoted_spread.mean()), 4),
                "market_spread_mean": round(float(market_spread.mean()), 4),
                "spread_ratio_mean": round(float(ratio.mean()), 2),
            }

        return pd.DataFrame(rows)

    def fill_quality(self) -> pd.DataFrame:
        """Per-fill analysis showing fill distance from best bid/ask and spread capture.

        Returns a DataFrame with one row per fill:
        - side, fill_price, filled_qty, fee
        - best_bid, best_ask at time of fill
        - distance: how far the fill was from the best level (positive = behind)
        - mid_at_fill: mid price at fill time
        - spread_capture: signed_qty * (mid - fill_price)
        """
        fills = self._get_fills()
        market_df = self.results.market_records_df()

        if fills.empty or market_df.empty:
            return pd.DataFrame()

        fills_reset = fills[["side", "fill_price", "filled_qty", "fee"]].reset_index()
        fills_reset["timestamp_event"] = fills_reset["timestamp_event"].astype(np.int64)
        market_reset = market_df[["best_bid_price", "best_ask_price", "mid_price"]].reset_index()
        market_reset["timestamp"] = market_reset["timestamp"].astype(np.int64)

        merged = pd.merge_asof(
            fills_reset.sort_values("timestamp_event"),
            market_reset.sort_values("timestamp"),
            left_on="timestamp_event",
            right_on="timestamp",
            direction="forward",
        )

        # Distance from best level
        merged["distance"] = np.where(
            merged["side"] == "Bid",
            merged["best_bid_price"] - merged["fill_price"],  # bid: positive = behind best bid
            merged["fill_price"] - merged["best_ask_price"],  # ask: positive = behind best ask
        )

        # Spread capture per fill
        merged["spread_capture"] = np.where(
            merged["side"] == "Bid",
            merged["filled_qty"] * (merged["mid_price"] - merged["fill_price"]),
            -merged["filled_qty"] * (merged["mid_price"] - merged["fill_price"]),
        )

        merged["timestamp_event"] = pd.to_datetime(merged["timestamp_event"])
        return merged[["timestamp_event", "side", "fill_price", "filled_qty", "fee",
                        "best_bid_price", "best_ask_price", "mid_price",
                        "distance", "spread_capture"]].set_index("timestamp_event")

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
            self.plot_pnl_decomposition(),
            self.plot_position(),
            self.plot_drawdown(),
            self.plot_spread(),
            self.plot_fees(),
            self.plot_quote_vs_spread(),
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
