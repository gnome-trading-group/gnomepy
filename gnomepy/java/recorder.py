from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy.java.statics import Scales


class BacktestResults:
    """Results from a backtest run, wrapping the Java BacktestRecorder.

    Provides efficient conversion of Java record lists into numpy arrays
    and pandas DataFrames for analysis.

    Usage:
        results = backtest.run()
        print(results.summary())
        df = results.execution_records_df()
        pnl = results.compute_pnl()
    """

    @property
    def PRICE_SCALE(self):
        return Scales.PRICE

    @property
    def SIZE_SCALE(self):
        return Scales.SIZE

    def __init__(self, java_recorder):
        self._java = java_recorder

    @property
    def market_record_count(self) -> int:
        return int(self._java.getMarketRecordCount())

    @property
    def execution_record_count(self) -> int:
        return int(self._java.getExecutionRecordCount())

    def market_records_df(self, scale_prices: bool = True) -> pd.DataFrame:
        """Convert Java market records to a pandas DataFrame.

        Args:
            scale_prices: If True, divide price fields by PRICE_SCALE
                to get human-readable values.

        Returns:
            DataFrame indexed by timestamp with market data columns.
        """
        records = self._java.getMarketRecords()
        if records.size() == 0:
            return pd.DataFrame()

        rows = []
        for r in records:
            rows.append({
                "timestamp": int(r.timestamp()),
                "exchange_id": int(r.exchangeId()),
                "security_id": int(r.securityId()),
                "best_bid_price": int(r.bestBidPrice()),
                "best_ask_price": int(r.bestAskPrice()),
                "best_bid_size": int(r.bestBidSize()),
                "best_ask_size": int(r.bestAskSize()),
                "last_trade_price": int(r.lastTradePrice()),
                "last_trade_size": int(r.lastTradeSize()),
                "sequence_number": int(r.sequenceNumber()),
            })

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        if scale_prices:
            for col in ["best_bid_price", "best_ask_price", "last_trade_price"]:
                df[col] = df[col] / self.PRICE_SCALE
            for col in ["best_bid_size", "best_ask_size", "last_trade_size"]:
                df[col] = df[col] / self.SIZE_SCALE

        return df

    def execution_records_df(self, scale_prices: bool = True) -> pd.DataFrame:
        """Convert Java execution records to a pandas DataFrame.

        Args:
            scale_prices: If True, divide price fields by PRICE_SCALE.

        Returns:
            DataFrame indexed by timestamp_event with execution columns.
        """
        records = self._java.getExecutionRecords()
        if records.size() == 0:
            return pd.DataFrame()

        rows = []
        for r in records:
            rows.append({
                "timestamp_event": int(r.timestampEvent()),
                "timestamp_recv": int(r.timestampRecv()),
                "exchange_id": int(r.exchangeId()),
                "security_id": int(r.securityId()),
                "client_oid": str(r.clientOid()),
                "side": str(r.side()),
                "exec_type": str(r.execType()),
                "order_status": str(r.orderStatus()),
                "filled_qty": int(r.filledQty()),
                "fill_price": int(r.fillPrice()),
                "cumulative_qty": int(r.cumulativeQty()),
                "leaves_qty": int(r.leavesQty()),
                "fee": float(r.fee()),
            })

        df = pd.DataFrame(rows)
        df["timestamp_event"] = pd.to_datetime(df["timestamp_event"])
        df["timestamp_recv"] = pd.to_datetime(df["timestamp_recv"])
        df = df.set_index("timestamp_event")

        if scale_prices:
            df["fill_price"] = df["fill_price"] / self.PRICE_SCALE
            df["fee"] = df["fee"] / (self.PRICE_SCALE * self.SIZE_SCALE)
            for col in ["filled_qty", "cumulative_qty", "leaves_qty"]:
                df[col] = df[col] / self.SIZE_SCALE

        return df

    def compute_pnl(self, scale_prices: bool = True) -> pd.DataFrame:
        """Compute PnL by merging market and execution data.

        Produces a time-series DataFrame with columns:
        - price: mid price from market data
        - quantity: net position (cumulative fills)
        - fee: cumulative fees
        - holding_pnl: PnL from price movement on existing position
        - trade_pnl: PnL from spread capture on new fills
        - pnl: total PnL (holding + trade - fees)
        - nmv: net market value (quantity * price)

        Returns:
            DataFrame indexed by timestamp.
        """
        market_df = self.market_records_df(scale_prices=scale_prices)
        exec_df = self.execution_records_df(scale_prices=scale_prices)

        if market_df.empty:
            return pd.DataFrame()

        # Compute mid price from best bid/ask
        df = market_df[["best_bid_price", "best_ask_price"]].copy()
        valid_book = (df["best_bid_price"] > 0) & (df["best_ask_price"] > 0)
        df["price"] = np.where(
            valid_book,
            (df["best_bid_price"] + df["best_ask_price"]) / 2.0,
            np.nan,
        )
        df["price"] = df["price"].ffill().bfill()
        df = df[["price"]].copy()

        # Merge execution data
        df["quantity"] = 0.0
        df["fee"] = 0.0
        df["fill_price"] = 0.0

        if not exec_df.empty:
            # Build net position from fills using side
            fills = exec_df[exec_df["exec_type"].isin(["FILL", "PARTIAL_FILL"])].copy()
            if not fills.empty:
                # Compute signed fill qty: Bid = +, Ask = -
                fills["signed_qty"] = fills.apply(
                    lambda r: r["filled_qty"] if r["side"] == "Bid" else -r["filled_qty"],
                    axis=1,
                )

                # Assign fills to nearest market timestamp
                market_times = df.index.asi8
                net_position = 0.0
                for _, fill in fills.iterrows():
                    fill_ns = fill.name.value
                    pos = np.searchsorted(market_times, fill_ns)
                    if pos >= len(market_times):
                        pos = len(market_times) - 1
                    elif pos > 0:
                        before = market_times[pos - 1]
                        after = market_times[pos]
                        if abs(fill_ns - before) < abs(fill_ns - after):
                            pos = pos - 1
                    df.iloc[pos, df.columns.get_loc("fee")] += fill["fee"]
                    df.iloc[pos, df.columns.get_loc("fill_price")] = fill["fill_price"]
                    net_position += fill["signed_qty"]
                    df.iloc[pos, df.columns.get_loc("quantity")] = net_position

        # Forward-fill position
        df["quantity"] = df["quantity"].replace(0, np.nan).ffill().fillna(0)
        df["fee"] = df["fee"].cumsum()

        # Compute PnL components
        prev_qty = df["quantity"].shift(1).fillna(0)
        prev_price = df["price"].shift(1)
        price_change = df["price"] - prev_price

        df["holding_pnl"] = (prev_qty * price_change).fillna(0)
        qty_change = df["quantity"] - prev_qty
        df["trade_pnl"] = (qty_change * (df["price"] - df["fill_price"].replace(0, np.nan).ffill().fillna(df["price"]))).fillna(0)
        df["pnl"] = (df["holding_pnl"] + df["trade_pnl"]).cumsum() - df["fee"]
        df["nmv"] = df["quantity"] * df["price"]

        return df

    def summary(self) -> dict:
        """Summary statistics for the backtest run.

        Returns:
            Dictionary with key performance metrics.
        """
        pnl_df = self.compute_pnl()
        exec_df = self.execution_records_df()

        result = {
            "market_records": self.market_record_count,
            "execution_records": self.execution_record_count,
        }

        if not pnl_df.empty:
            result.update({
                "total_pnl": float(pnl_df["pnl"].iloc[-1]) if len(pnl_df) > 0 else 0.0,
                "total_fees": float(pnl_df["fee"].iloc[-1]) if len(pnl_df) > 0 else 0.0,
                "max_nmv": float(pnl_df["nmv"].abs().max()),
                "final_quantity": float(pnl_df["quantity"].iloc[-1]) if len(pnl_df) > 0 else 0.0,
            })

        if not exec_df.empty:
            fills = exec_df[exec_df["exec_type"].isin(["FILL", "PARTIAL_FILL"])]
            result.update({
                "num_fills": len(fills),
                "num_cancels": len(exec_df[exec_df["exec_type"] == "CANCEL"]),
                "num_rejects": len(exec_df[exec_df["exec_type"] == "REJECT"]),
            })

        return result

    def __repr__(self) -> str:
        return (
            f"BacktestResults(market_records={self.market_record_count}, "
            f"execution_records={self.execution_record_count})"
        )
