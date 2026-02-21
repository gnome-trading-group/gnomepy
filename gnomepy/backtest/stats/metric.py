from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any, Optional, List


class Metric(ABC):
    """
    A base class for computing a strategy's performance metrics. Implementing a custom metric class derived from this
    base class enables the computation of the custom metric in the :class:`Stats` and displays the summary.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def _compute(self, df: pd.DataFrame, ctx: dict):
        raise NotImplementedError

    def compute(self, df: pd.DataFrame, ctx: dict[str, Any]) -> Any:
        # Check if already computed
        if self.name in ctx:
            return ctx[self.name]

        # Compute the metric
        try:
            value = self._compute(df, ctx)
            ctx[self.name] = value
            return value
        except Exception as e:
            raise RuntimeError(f"Error computing metric '{self.name}': {str(e)}") from e


class TotalPnL(Metric):
    """
    Total Profit and Loss.
    """
    def __init__(self, name: str = "total_pnl"):
        super().__init__(name)

    def _compute(self, df: pd.DataFrame, ctx: dict) -> np.floating:
        return df['pnl'].sum()


class TotalFees(Metric):
    """
    Total fees paid.
    """
    def __init__(self, name: str = "total_fees"):
        super().__init__(name)

    def _compute(self, df: pd.DataFrame, ctx: dict) -> np.floating:
        return df['fee'].sum()


class Sharpe(Metric):
    """
    Sharpe Ratio.
    """
    def __init__(self, name: str = "sr"):
        super().__init__(name)

    def _compute(self, df: pd.DataFrame, ctx: dict) -> np.floating:
        pnl = df['pnl']

        if pnl.std() == 0.0:
            return np.nan

        return pnl.mean() / pnl.std()


class MaxDrawdown(Metric):
    """
    Maximum Drawdown
    """
    def __init__(self, name: str = 'max_dd', book_size: float | None = None):
        super().__init__(name)
        self.book_size = book_size

    def _compute(self, df: pd.DataFrame, ctx: dict) -> np.floating:
        equity = df['pnl']

        max_equity = equity.cummax()
        dd = equity - max_equity

        if self.book_size is not None:
            dd /= self.book_size

        return abs(dd.min())


class WinRate(Metric):
    """
    Win Rate - percentage of profitable time steps.
    """
    def __init__(self, name: str = "win_rate"):
        super().__init__(name)

    def _compute(self, df: pd.DataFrame, ctx: dict) -> np.floating:
        pnl = df['pnl']
        if len(pnl) == 0:
            return np.nan
        return (pnl > 0).mean() * 100


class ReturnOnFees(Metric):
    """
    Return on fees - PnL divided by total fees.
    """
    def __init__(self, name: str = "return_on_fees"):
        super().__init__(name)

    def _compute(self, df: pd.DataFrame, ctx: dict) -> np.floating:
        pnl = TotalPnL().compute(df, ctx)
        total_fees = TotalFees().compute(df, ctx)
        
        if total_fees == 0:
            return np.inf if pnl > 0 else np.nan
        
        return pnl / total_fees


class Volatility(Metric):
    """
    Volatility of returns.
    """
    def __init__(self, name: str = "volatility"):
        super().__init__(name)

    def _compute(self, df: pd.DataFrame, ctx: dict) -> np.floating:
        pnl = df['pnl']
        if len(pnl) <= 1:
            return np.nan
        return pnl.std()


class TradeVolume(Metric):
    """
    Total trade volume - sum of absolute quantity changes for all executions.
    Calculates the total volume traded (in shares/units) across all execution events.
    """
    def __init__(self, name: str = "trade_volume"):
        super().__init__(name)

    def _compute(self, df: pd.DataFrame, ctx: dict) -> np.floating:
        # Check if we have execution events
        if 'event' not in df.columns:
            return 0.0
        
        from gnomepy.backtest.recorder import RecordType
        
        # Calculate quantity changes for all rows
        quantity_changes = df['quantity'].diff().abs()
        
        # Sum only the quantity changes that occur at execution events
        # This gives total trade volume (absolute value of all position changes from trades)
        trade_volume = quantity_changes[df['event'] == RecordType.EXECUTION].sum()
        
        # Handle first execution if it exists and has NaN change
        executions = df[df['event'] == RecordType.EXECUTION]
        if len(executions) > 0:
            first_exec_idx = executions.index[0]
            if pd.isna(quantity_changes.loc[first_exec_idx]):
                # First execution: use absolute quantity as trade size
                trade_volume += abs(executions['quantity'].iloc[0])
        
        return trade_volume


class NumberOfTrades(Metric):
    """
    Number of trades - count of execution events.
    """
    def __init__(self, name: str = "num_trades"):
        super().__init__(name)

    def _compute(self, df: pd.DataFrame, ctx: dict) -> np.floating:
        # Check if we have execution events
        if 'event' not in df.columns:
            return 0.0
        
        from gnomepy.backtest.recorder import RecordType
        
        # Count execution events
        num_trades = (df['event'] == RecordType.EXECUTION).sum()
        
        return float(num_trades)


DEFAULT_METRICS = [
    TotalPnL(),
    TotalFees(),
    Sharpe(),
    MaxDrawdown(),
    Volatility(),
    WinRate(),
    ReturnOnFees(),
    TradeVolume(),
    NumberOfTrades(),
]
