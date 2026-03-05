from __future__ import annotations

from typing import Any, Optional, Union, Literal
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from numpy.typing import NDArray

from gnomepy.backtest.stats.metric import Metric, DEFAULT_METRICS
from gnomepy.backtest.stats.utils import resample, partition, compute_metrics, IntervalType
from gnomepy.backtest.recorder import RecordType


PlotValueType = Literal['nmv', 'quantity', 'pnl', 'fee']


@dataclass
class PlotConfig:
    """Configuration for plotting."""
    plots: list[PlotValueType]
    figsize: tuple[int, int] = (10, 12)
    style: Optional[dict] = None
    save_path: Optional[str] = None


class BaseRecord(ABC):
    """Pandas-oriented wrapper around a structured event array.

    Provides convenience methods to prepare data, compute metrics, and produce
    resampled or partitioned views.
    """

    def __init__(self, arr: NDArray):
        self.arr = arr
        self.df = pd.DataFrame(arr)
        self.prepare()

    def prepare(self):
        """Prepare the record for analysis."""
        self.df = self._prepare_dataframe(self.df)

    @abstractmethod
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare the DataFrame for analysis."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_dtype(cls) -> np.dtype:
        """Return the numpy dtype for this recorder's records."""
        raise NotImplementedError

    def stats(
            self,
            metrics: list[Metric] | None = None,
            interval: IntervalType | None = None,
            frequency: str | None = None,
            **kwargs
    ) -> 'Stats':
        """Compute metrics over the entire dataset and optional partitions.

        Parameters
        ----------
        metrics : list[Metric], optional
            Metrics to compute; defaults to `DEFAULT_METRICS`.
        interval : IntervalType | optional
            If provided, also compute metrics for each partitioned slice.
        frequency : str, optional
            If provided, resample the data to this frequency prior to evaluation.

        Returns
        -------
        Stats
            A `Stats` object containing the original (or resampled) data and
            a list of context dictionaries with metric results.
        """
        if metrics is None:
            metrics = DEFAULT_METRICS

        # Work with a copy to avoid side effects
        df = self.df.copy()

        if frequency is not None:
            df = resample(df, frequency)
            df = self._prepare_dataframe(df)

        partitions = [df]

        if interval is not None:
            partitions.extend(partition(df, interval))

        stats = [compute_metrics(sub_df, metrics) for sub_df in partitions]

        return Stats(df, stats, **kwargs)


class IntentRecord(BaseRecord):
    """Pandas-oriented wrapper around a structured intent array.
    
    Similar to Record, but specifically for trading intent events
    (side, confidence, price, flatten) tracked when intents are generated.
    
    Provides convenience methods to prepare data and analyze intent behavior.
    """

    @classmethod
    def get_dtype(cls) -> np.dtype:
        return np.dtype(
            [
                ('timestamp', 'i8'),
                ('side', 'U1'),  # 'B' or 'S'/'A'
                ('confidence', 'f8'),
                ('price', 'f8'),
                ('flatten', 'i8'),  # 0 or 1
            ],
            align=True
        )
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame without mutating the original."""
        df = df.copy()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        # Forward fill prices and confidence for continuity
        df['price'] = df['price'].replace(0.0, np.nan).ffill().bfill()
        df['confidence'] = df['confidence'].ffill().bfill()
        df['flatten'] = df['flatten'].fillna(0).astype(bool)
        
        # Calculate derived metrics
        df['is_buy'] = df['side'] == 'B'
        df['is_sell'] = df['side'].isin(['S', 'A'])
        
        return df


class MarketRecord(BaseRecord):
    """Pandas-oriented wrapper around a structured market array.
    
    Similar to Record, but specifically for market events
    (price, quantity, fee) tracked when market data is received.
    """
    
    @classmethod
    def get_dtype(cls) -> np.dtype:
        return np.dtype(
            [
                ('event', 'i8'),
                ('timestamp', 'i8'),
                ('price', 'f8'),
                ('fill_price', 'f8'),
                ('quantity', 'f8'),
                ('fee', 'f8'),
            ],
            align=True
        )

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame without mutating the original."""
        df = df.copy()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

        df['price'] = df['price'].replace(0.0, np.nan).ffill().bfill()
        df['quantity'] = df['quantity'].ffill().bfill()
        df['fee'] = df['fee'].fillna(0.0)

        df['trade_price'] = np.where(
            (df['event'] == RecordType.EXECUTION) & (df['fill_price'] > 0),
            df['fill_price'],
            df['price'],
        )

        # TODO: some logs have the same event timestamp... how to fix properly?
        df = df.groupby(level=0).agg(
            {"event": "mean", "price": "mean", "trade_price": "mean", "quantity": "mean", "fee": "sum"}
        )
        
        # Derived columns
        previous_quantity = df['quantity'].shift(1).fillna(0.0)
        previous_price = df['price'].shift(1)
        price_change = df['price'] - previous_price
        quantity_change = df['quantity'] - previous_quantity

        # Holding PnL: price movement on prior position (shares * $/share = $)
        df['holding_pnl'] = (previous_quantity * price_change).fillna(0.0)

        # Trade PnL: spread capture on new fills (shares * $/share = $)
        df['trade_pnl'] = (quantity_change * (df['price'] - df['trade_price'])).fillna(0.0)

        # Composite PnL
        df['pnl_wo_fee'] = df['holding_pnl'] + df['trade_pnl']
        df['pnl'] = df['pnl_wo_fee'] - df['fee']

        # Net market value
        df['nmv'] = df['quantity'] * df['price']

        return df

class Stats:
    """Container for computed statistics and source data with convenience views."""

    def __init__(self, data: pd.DataFrame, stats: list[dict[str, Any]], **kwargs):
        self._validate_data(data)
        self._validate_stats(stats)
        self.data = data
        self.stats = stats
        self.kwargs = kwargs
        self._cache: dict[str, Any] = {}
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input DataFrame."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        
        if data.empty:
            raise ValueError("DataFrame cannot be empty")
        
        required_columns = ['nmv', 'price', 'quantity', 'fee']
        missing = set(required_columns) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def _validate_stats(self, stats: list[dict[str, Any]]) -> None:
        """Validate stats list."""
        if not isinstance(stats, list):
            raise TypeError("stats must be a list")
        
        if not stats:
            raise ValueError("stats list cannot be empty")
        
        for i, stat in enumerate(stats):
            if not isinstance(stat, dict):
                raise TypeError(f"stats[{i}] must be a dictionary")

    def summary(self, format: str = 'dataframe') -> Union[pd.DataFrame, dict, list[dict[str, Any]]]:
        """Return summary in different formats.
        
        Parameters
        ----------
        format : str, default 'dataframe'
            Format to return: 'dataframe', 'dict', or 'list'
            
        Returns
        -------
        Union[pd.DataFrame, dict, list[dict[str, Any]]]
            Summary in the requested format
        """
        if format == 'dataframe':
            return pd.DataFrame(self.stats)
        elif format == 'dict':
            return self.stats[0] if len(self.stats) == 1 else self.stats
        elif format == 'list':
            return self.stats
        else:
            raise ValueError(f"Unknown format: {format}. Use 'dataframe', 'dict', or 'list'")
    
    def get_metric(self, metric_name: str, partition: int = 0) -> Any:
        """Get specific metric value.
        
        Parameters
        ----------
        metric_name : str
            Name of the metric to retrieve
        partition : int, default 0
            Partition index (0 for overall stats)
            
        Returns
        -------
        Any
            The metric value
        """
        if partition >= len(self.stats):
            raise IndexError(f"Partition {partition} out of range (max: {len(self.stats) - 1})")
        
        return self.stats[partition].get(metric_name)
    
    def get_performance_summary(self) -> dict[str, Any]:
        """Get key performance indicators."""
        if not self.stats:
            return {}
        
        main_stats = self.stats[0]  # Overall stats
        return {
            'total_pnl': main_stats.get('pnl', 0),
            'sharpe_ratio': main_stats.get('sr', 0),
            'max_drawdown': main_stats.get('max_dd', 0),
            'total_fees': self.data['fee'].sum(),
            'trading_days': len(self.data),
            'start_date': self.data.index.min(),
            'end_date': self.data.index.max(),
        }
    
    def export_to_csv(self, path: str, include_data: bool = False) -> None:
        """Export summary to CSV.
        
        Parameters
        ----------
        path : str
            Path to save the CSV file
        include_data : bool, default False
            Whether to also export the underlying data
        """
        summary = self.summary('dataframe')
        summary.to_csv(path)
        
        if include_data:
            data_path = path.replace('.csv', '_data.csv')
            self.data.to_csv(data_path)
    
    def compare_with(self, other: 'Stats', metrics: Optional[list[str]] = None) -> pd.DataFrame:
        """Compare metrics with another Stats instance.
        
        Parameters
        ----------
        other : Stats
            Another Stats instance to compare with
        metrics : list[str], optional
            Specific metrics to compare. If None, compares all available metrics.
            
        Returns
        -------
        pd.DataFrame
            Comparison table with both instances' metrics
        """
        if not isinstance(other, Stats):
            raise TypeError("other must be a Stats instance")
        
        # Get metrics to compare
        if metrics is None:
            metrics = set(self.stats[0].keys()) & set(other.stats[0].keys())
        else:
            metrics = set(metrics)
        
        comparison_data = []
        for metric in metrics:
            comparison_data.append({
                'metric': metric,
                'self': self.get_metric(metric),
                'other': other.get_metric(metric),
                'difference': self.get_metric(metric) - other.get_metric(metric)
            })
        
        return pd.DataFrame(comparison_data)

    def plot(self, 
             config: Optional[PlotConfig] = None,
             price_as_ret: bool = False) -> plt.Figure:
        """Enhanced plotting with flexible configuration.

        Parameters
        ----------
        config : PlotConfig, optional
            Configuration object for plotting. If None, uses default plots.
        price_as_ret : bool, default False
            If True and `book_size` is supplied via external context, plot price
            as returns; otherwise plot absolute values with price on a twin axis.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure (closed; safe for embedding or saving).
        """
        if config is None:
            config = PlotConfig(plots=['nmv', 'quantity', 'pnl', 'fee'])
        
        # Validate plots
        available_plots = ['nmv', 'quantity', 'pnl', 'fee', 'price']
        invalid_plots = set(config.plots) - set(available_plots)
        if invalid_plots:
            raise ValueError(f"Invalid plots: {invalid_plots}. Available: {available_plots}")
        
        n_plots = len(config.plots)
        if n_plots == 0:
            raise ValueError("At least one plot must be specified")

        from matplotlib import pyplot as plt

        fig, axes = plt.subplots(n_plots, 1, sharex=True)
        if n_plots == 1:
            axes = [axes]
        
        fig.subplots_adjust(hspace=0)
        fig.set_size_inches(*config.figsize)

        entire_df = self.data.reset_index()
        book_size = self.kwargs.get('book_size')
        
        # Plot each requested plot type
        for i, plot_type in enumerate(config.plots):
            ax = axes[i]
            
            if plot_type == 'nmv':
                self._plot_nmv(ax, entire_df, book_size)
            elif plot_type == 'quantity':
                self._plot_quantity(ax, entire_df)
            elif plot_type == 'pnl':
                self._plot_pnl(ax, entire_df, book_size, price_as_ret)
            elif plot_type == 'fee':
                self._plot_fee(ax, entire_df)
            elif plot_type == 'price':
                self._plot_price(ax, entire_df)
        
        # Save if requested
        if config.save_path:
            fig.savefig(config.save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return fig
    
    def _plot_nmv(self, ax, df, book_size):
        """Plot NMV (Net Market Value)."""
        nmv = df['nmv']

        if book_size is not None:
            ax.plot(df['timestamp'], nmv / book_size * 100, label='NMV')
            ax.set_ylabel('NMV % Of Book Size')
        else:
            ax.plot(df['timestamp'], nmv, label='NMV')
            ax.set_ylabel('NMV')
        
        # Add price on twin axis
        ax2 = ax.twinx()
        ax2.plot(df['timestamp'], df['price'], 'black', alpha=0.3, label='Price')
        ax2.set_ylabel('Price')
        
        # Combine legends
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc='best')
        
        ax.grid()
    
    def _plot_quantity(self, ax, df):
        """Plot quantity."""
        ax.plot(df['timestamp'], df['quantity'], label='Quantity', color='blue')
        ax.set_ylabel('Quantity')
        
        # Add price on twin axis
        ax2 = ax.twinx()
        ax2.plot(df['timestamp'], df['price'], 'black', alpha=0.3, label='Price')
        ax2.set_ylabel('Price')
        
        # Combine legends
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc='best')
        
        ax.grid()
    
    def _plot_pnl(self, ax, df, book_size, price_as_ret):
        """Plot PnL."""
        cpnl = df['pnl'].cumsum()
        cpnl_wo_fee = df['pnl_wo_fee'].cumsum()

        if price_as_ret:
            if book_size is not None:
                ax.plot(df['timestamp'], cpnl_wo_fee / book_size * 100, label='Cumulative Ret (%) w/o Fee')
                ax.plot(df['timestamp'], cpnl / book_size * 100, label='Cumulative Ret (%)')
                ax.set_ylabel('Cumulative Returns (%)')
            else:
                raise ValueError("Book size is expected if plotting return.")
        else:
            ax.plot(df['timestamp'], cpnl_wo_fee, label='Cumulative PnL w/o Fee')
            ax.plot(df['timestamp'], cpnl, label='Cumulative PnL')
            ax.set_ylabel('Cumulative PnL')
        
        # Add price on twin axis
        ax2 = ax.twinx()
        ax2.plot(df['timestamp'], df['price'], 'black', alpha=0.3, label='Price')
        ax2.set_ylabel('Price')
        
        # Combine legends
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc='best')
        
        ax.grid()
    
    def _plot_fee(self, ax, df):
        """Plot cumulative fee."""
        cumulative_fee = df['fee'].cumsum()
        ax.plot(df['timestamp'], cumulative_fee, label='Cumulative Fee', color='red')
        ax.set_ylabel('Cumulative Fee')
        
        # Add price on twin axis
        ax2 = ax.twinx()
        ax2.plot(df['timestamp'], df['price'], 'black', alpha=0.3, label='Price')
        ax2.set_ylabel('Price')
        
        # Combine legends
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc='best')
        
        ax.grid()
    
    def _plot_price(self, ax, df):
        """Plot price."""
        ax.plot(df['timestamp'], df['price'], label='Price', color='black', alpha=0.7)
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid()
