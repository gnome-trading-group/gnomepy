from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy.java.statics import Scales


class BacktestResults:
    """Data access layer for backtest recorder output.

    Converts columnar Java arrays into cached pandas DataFrames.
    For analysis and reporting, use BacktestReport(results).

    Usage:
        results = backtest.run()
        market_df = results.market_records_df()
        exec_df = results.execution_records_df()
        intent_df = results.intent_records_df()
        results.save("/tmp/my_backtest")
    """

    @property
    def PRICE_SCALE(self):
        return Scales.PRICE

    @property
    def SIZE_SCALE(self):
        return Scales.SIZE

    def __init__(self, java_recorder):
        self._java = java_recorder
        self._cached_market_df = None
        self._cached_exec_df = None
        self._cached_intent_df = None

    @property
    def market_record_count(self) -> int:
        return int(self._java.getMarketRecordCount())

    @property
    def execution_record_count(self) -> int:
        return int(self._java.getExecutionRecordCount())

    @property
    def intent_record_count(self) -> int:
        return int(self._java.getIntentRecordCount())

    def market_records_df(self, scale_prices: bool = True) -> pd.DataFrame:
        """Convert Java market record arrays to a pandas DataFrame."""
        if self._cached_market_df is not None:
            return self._cached_market_df

        n = self.market_record_count
        if n == 0:
            return pd.DataFrame()

        j = self._java
        df = pd.DataFrame({
            "timestamp": np.array(j.getMarketTimestamps()[:n], dtype=np.int64),
            "exchange_id": np.array(j.getMarketExchangeIds()[:n], dtype=np.int32),
            "security_id": np.array(j.getMarketSecurityIds()[:n], dtype=np.int64),
            "best_bid_price": np.array(j.getMarketBestBidPrices()[:n], dtype=np.int64),
            "best_ask_price": np.array(j.getMarketBestAskPrices()[:n], dtype=np.int64),
            "best_bid_size": np.array(j.getMarketBestBidSizes()[:n], dtype=np.int64),
            "best_ask_size": np.array(j.getMarketBestAskSizes()[:n], dtype=np.int64),
            "mid_price": np.array(j.getMarketMidPrices()[:n], dtype=np.int64),
            "spread": np.array(j.getMarketSpreads()[:n], dtype=np.int64),
            "last_trade_price": np.array(j.getMarketLastTradePrices()[:n], dtype=np.int64),
            "last_trade_size": np.array(j.getMarketLastTradeSizes()[:n], dtype=np.int64),
        })

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        if scale_prices:
            for col in ["best_bid_price", "best_ask_price", "mid_price", "spread",
                        "last_trade_price"]:
                df[col] = df[col] / self.PRICE_SCALE
            for col in ["best_bid_size", "best_ask_size", "last_trade_size"]:
                df[col] = df[col] / self.SIZE_SCALE

        self._cached_market_df = df
        return df

    def execution_records_df(self, scale_prices: bool = True) -> pd.DataFrame:
        """Convert Java execution record arrays to a pandas DataFrame."""
        if self._cached_exec_df is not None:
            return self._cached_exec_df

        n = self.execution_record_count
        if n == 0:
            return pd.DataFrame()

        j = self._java
        df = pd.DataFrame({
            "timestamp_event": np.array(j.getExecTimestampEvents()[:n], dtype=np.int64),
            "timestamp_recv": np.array(j.getExecTimestampRecvs()[:n], dtype=np.int64),
            "exchange_id": np.array(j.getExecExchangeIds()[:n], dtype=np.int32),
            "security_id": np.array(j.getExecSecurityIds()[:n], dtype=np.int32),
            "strategy_id": np.array(j.getExecStrategyIds()[:n], dtype=np.int32),
            "client_oid": list(j.getExecClientOids()[:n]),
            "side": list(j.getExecSides()[:n]),
            "exec_type": list(j.getExecExecTypes()[:n]),
            "filled_qty": np.array(j.getExecFilledQtys()[:n], dtype=np.int64),
            "fill_price": np.array(j.getExecFillPrices()[:n], dtype=np.int64),
            "order_price": np.array(j.getExecOrderPrices()[:n], dtype=np.int64),
            "order_size": np.array(j.getExecOrderSizes()[:n], dtype=np.int64),
            "fee": np.array(j.getExecFees()[:n], dtype=np.float64),
        })

        df["timestamp_event"] = pd.to_datetime(df["timestamp_event"])
        df["timestamp_recv"] = pd.to_datetime(df["timestamp_recv"])
        df = df.set_index("timestamp_event")

        if scale_prices:
            for col in ["fill_price", "order_price"]:
                df[col] = df[col] / self.PRICE_SCALE
            for col in ["filled_qty", "order_size"]:
                df[col] = df[col] / self.SIZE_SCALE
            df["fee"] = df["fee"] / (self.PRICE_SCALE * self.SIZE_SCALE)

        self._cached_exec_df = df
        return df

    def intent_records_df(self, scale_prices: bool = True) -> pd.DataFrame:
        """Convert Java intent record arrays to a pandas DataFrame."""
        if self._cached_intent_df is not None:
            return self._cached_intent_df

        n = self.intent_record_count
        if n == 0:
            return pd.DataFrame()

        j = self._java
        df = pd.DataFrame({
            "timestamp": np.array(j.getIntentTimestamps()[:n], dtype=np.int64),
            "exchange_id": np.array(j.getIntentExchangeIds()[:n], dtype=np.int32),
            "security_id": np.array(j.getIntentSecurityIds()[:n], dtype=np.int64),
            "strategy_id": np.array(j.getIntentStrategyIds()[:n], dtype=np.int32),
            "bid_price": np.array(j.getIntentBidPrices()[:n], dtype=np.int64),
            "bid_size": np.array(j.getIntentBidSizes()[:n], dtype=np.int64),
            "ask_price": np.array(j.getIntentAskPrices()[:n], dtype=np.int64),
            "ask_size": np.array(j.getIntentAskSizes()[:n], dtype=np.int64),
            "take_side": list(j.getIntentTakeSides()[:n]),
            "take_size": np.array(j.getIntentTakeSizes()[:n], dtype=np.int64),
            "take_limit_price": np.array(j.getIntentTakeLimitPrices()[:n], dtype=np.int64),
        })

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        if scale_prices:
            for col in ["bid_price", "ask_price", "take_limit_price"]:
                df[col] = df[col] / self.PRICE_SCALE
            for col in ["bid_size", "ask_size", "take_size"]:
                df[col] = df[col] / self.SIZE_SCALE

        self._cached_intent_df = df
        return df

    def save(self, directory: str | Path) -> None:
        """Save all recorder data to Parquet files in the given directory.

        Creates three files:
          - market_records.parquet
          - execution_records.parquet
          - intent_records.parquet
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        market_df = self.market_records_df()
        if not market_df.empty:
            market_df.to_parquet(directory / "market_records.parquet")

        exec_df = self.execution_records_df()
        if not exec_df.empty:
            exec_df.to_parquet(directory / "execution_records.parquet")

        intent_df = self.intent_records_df()
        if not intent_df.empty:
            intent_df.to_parquet(directory / "intent_records.parquet")

    def __repr__(self) -> str:
        return (
            f"BacktestResults(market_records={self.market_record_count}, "
            f"execution_records={self.execution_record_count}, "
            f"intent_records={self.intent_record_count})"
        )
