import pandas as pd
from matplotlib import pyplot as plt

from typing import Any, Optional, Union

from gnomepy.backtest.stats.stats import Stats, PlotConfig
from gnomepy.backtest.stats.metric import Metric, DEFAULT_METRICS
from gnomepy.backtest.stats.utils import resample, compute_metrics_for_intervals, IntervalType


class StrategyStats:

    def __init__(
            self,
            per_listing_stats: dict[int, Stats],
            metrics: list[Metric] | None = None,
            frequency: str | None = None,
            interval: IntervalType | None = None,
    ):
        """Aggregate per-listing `Stats` and compute portfolio-level analytics.

        Parameters
        ----------
        per_listing_stats : dict[int, Stats]
            Mapping of listing id to its corresponding `Stats` (with data).
        metrics : list[Metric] | None, optional
            Metrics to compute at the portfolio level; defaults to `DEFAULT_METRICS`.
        frequency : str | None, optional
            If provided, resample each listing's data to this pandas frequency
            string before aggregation (e.g., '1min', '5s').
        interval : IntervalType | None, optional
            Optional interval partitioning used when computing metrics.
        """
        self._validate_per_listing_stats(per_listing_stats)
        self.per_listing_stats = per_listing_stats
        self.frequency = frequency
        self.data = self._build_portfolio_frame()
        self.stats = self._compute_stats(self.data, metrics, interval)
        self._cache: dict[str, Any] = {}
    
    def _validate_per_listing_stats(self, per_listing_stats: dict[int, Stats]) -> None:
        """Validate per-listing stats input."""
        if not isinstance(per_listing_stats, dict):
            raise TypeError("per_listing_stats must be a dictionary")
        
        if not per_listing_stats:
            raise ValueError("per_listing_stats cannot be empty")
        
        for listing_id, stats in per_listing_stats.items():
            if not isinstance(listing_id, int):
                raise TypeError(f"Listing ID must be int, got {type(listing_id)}")
            if not isinstance(stats, Stats):
                raise TypeError(f"Stats must be Stats instance, got {type(stats)}")

    @staticmethod
    def _compute_pnl_series(df: pd.DataFrame) -> pd.Series:
        """Compute per-period PnL for a single listing frame."""
        pnl = df["nmv"].shift() * df["price"].pct_change()
        return pnl.fillna(0.0)

    def _build_portfolio_frame(self) -> pd.DataFrame:
        """Construct a portfolio-level DataFrame from per-listing `Stats.data`."""
        nmv_frames: list[pd.Series] = []
        pnl_frames: list[pd.Series] = []
        fee_frames: list[pd.Series] = []
        indices: list[pd.Index] = []

        for listing_id, stats in self.per_listing_stats.items():
            df = stats.data

            if self.frequency is not None:
                df = resample(df, self.frequency)

            nmv_series = df["nmv"].rename(listing_id)
            pnl_series = self._compute_pnl_series(df).rename(listing_id)
            fee_series = df["fee"].rename(listing_id)

            nmv_frames.append(nmv_series)
            pnl_frames.append(pnl_series)
            fee_frames.append(fee_series)
            indices.append(df.index)

        if len(nmv_frames) == 0:
            return pd.DataFrame(columns=["nmv", "pnl", "fee"]).set_index(pd.DatetimeIndex([], name="timestamp"))

        # Build a common union index across listings
        common_index = indices[0]
        for idx in indices[1:]:
            common_index = common_index.union(idx)
        common_index = common_index.sort_values()

        # Reindex: nmv forward-filled (position continuity), pnl zeros when missing
        nmv_aligned = [s.reindex(common_index).ffill() for s in nmv_frames]
        pnl_aligned = [s.reindex(common_index).fillna(0.0) for s in pnl_frames]
        fee_aligned = [s.reindex(common_index).fillna(0.0) for s in fee_frames]

        nmv = pd.concat(nmv_aligned, axis=1).sum(axis=1, min_count=1)
        pnl = pd.concat(pnl_aligned, axis=1).sum(axis=1, min_count=1)
        fee = pd.concat(fee_aligned, axis=1).sum(axis=1, min_count=1)

        combined = pd.DataFrame({
            "nmv": nmv,
            "pnl": pnl,
            "fee": fee,
        })

        combined.index.name = "timestamp"
        return combined.sort_index()

    @staticmethod
    def _compute_stats(
            df: pd.DataFrame,
            metrics: list[Metric] | None = None,
            interval: IntervalType | None = None
    ) -> list[dict[str, Any]]:
        """Compute metrics over the provided portfolio DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Portfolio-level DataFrame with at least `nmv` and `pnl` columns.
        metrics : list[Metric] | None, optional
            Metrics to compute; defaults to `DEFAULT_METRICS`.
        interval : IntervalType | None, optional
            If provided, compute metrics for each partition slice as well.
        """
        if metrics is None:
            metrics = DEFAULT_METRICS

        return compute_metrics_for_intervals(df, metrics, interval)

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
            'num_listings': len(self.per_listing_stats),
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

    def listing_summary(self) -> pd.DataFrame:
        """Return a summary DataFrame with per-listing metric rows."""
        frames: list[pd.DataFrame] = []

        for listing_id, stats in self.per_listing_stats.items():
            df = stats.summary().copy()
            df["listing_id"] = listing_id
            frames.append(df)

        if len(frames) == 0:
            return pd.DataFrame(columns=["listing_id"])

        return pd.concat(frames, ignore_index=True)

    def plot(self, 
             config: Optional[PlotConfig] = None,
             price_as_ret: bool = False, 
             book_size: float | None = None) -> plt.Figure:
        """Enhanced plotting with flexible configuration.

        Parameters
        ----------
        config : PlotConfig, optional
            Configuration object for plotting. If None, uses default plots.
        price_as_ret : bool, default False
            Preserved for API parity with record.Stats.plot; if True and
            `book_size` is provided, equity is shown as returns (%).
        book_size : float | None, default None
            If provided, scales equity and cumulative PnL for comparability.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure (closed; safe for embedding or saving).
        """
        if config is None:
            config = PlotConfig(plots=['nmv', 'pnl', 'fee'])
        
        # Validate plots
        available_plots = ['nmv', 'pnl', 'fee', 'price']
        invalid_plots = set(config.plots) - set(available_plots)
        if invalid_plots:
            raise ValueError(f"Invalid plots: {invalid_plots}. Available: {available_plots}")
        
        n_plots = len(config.plots)
        if n_plots == 0:
            raise ValueError("At least one plot must be specified")
        
        df = self.data.reset_index()

        # Handle empty data
        if df.empty:
            fig, axes = plt.subplots(n_plots, 1, sharex=True)
            if n_plots == 1:
                axes = [axes]
            fig.subplots_adjust(hspace=0)
            fig.set_size_inches(*config.figsize)
            plt.close()
            return fig

        fig, axes = plt.subplots(n_plots, 1, sharex=True)
        if n_plots == 1:
            axes = [axes]
        
        fig.subplots_adjust(hspace=0)
        fig.set_size_inches(*config.figsize)
        
        # Plot each requested plot type
        for i, plot_type in enumerate(config.plots):
            ax = axes[i]
            
            if plot_type == 'nmv':
                self._plot_nmv(ax, df, book_size, price_as_ret)
            elif plot_type == 'pnl':
                self._plot_pnl(ax, df, book_size, price_as_ret)
            elif plot_type == 'fee':
                self._plot_fee(ax, df)

        # Save if requested
        if config.save_path:
            fig.savefig(config.save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return fig
    
    def _plot_nmv(self, ax, df, book_size, price_as_ret):
        """Plot NMV (Net Market Value)."""
        nmv = df['nmv']
        
        if book_size is not None:
            if price_as_ret:
                ax.plot(df['timestamp'], nmv / book_size * 100, label='NMV')
                ax.set_ylabel('Cumulative Returns (%)')
            else:
                ax.plot(df['timestamp'], nmv / book_size * 100, label='Equity')
                ax.set_ylabel('Cumulative Returns (%)')
        else:
            ax.plot(df['timestamp'], nmv, label='NMV')
            ax.set_ylabel('NMV')

        ax.legend(loc='best')
        ax.grid()
    
    def _plot_pnl(self, ax, df, book_size, price_as_ret):
        """Plot PnL."""
        pnl_w_fee = df['pnl'] - df['fee']
        cpnl = pnl_w_fee.cumsum()
        
        if book_size is not None:
            if price_as_ret:
                ax.plot(df['timestamp'], cpnl / book_size * 100, label='Cum Ret (%)')
                ax.set_ylabel('Cumulative Returns (%)')
            else:
                ax.plot(df['timestamp'], cpnl / book_size, label='Cum PnL (% of book)')
                ax.set_ylabel('Cumulative PnL (% of book)')
        else:
            ax.plot(df['timestamp'], cpnl, label='Cumulative PnL')
            ax.set_ylabel('Cumulative PnL')

        ax.legend(loc='best')
        ax.grid()
    
    def _plot_fee(self, ax, df):
        """Plot cumulative fee."""
        cumulative_fee = df['fee'].cumsum()
        ax.plot(df['timestamp'], cumulative_fee, label='Cumulative Fee', color='red')
        ax.set_ylabel('Cumulative Fee')

        ax.legend(loc='best')
        ax.grid()

    # Backward compatibility method
    def plot_legacy(self, price_as_ret: bool = False, book_size: float | None = None) -> plt.Figure:
        """Legacy plotting method for backward compatibility.
        
        This method maintains the original 4-subplot layout.
        """
        config = PlotConfig(plots=['nmv', 'pnl', 'fee'])
        return self.plot(config, price_as_ret, book_size)
