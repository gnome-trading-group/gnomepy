from gnomepy.data.client import MarketDataClient
from gnomepy.data.types import SchemaType
from gnomepy.research_old.archive.strategy_old import Strategy
from gnomepy.research_old.strategy import *
from gnomepy.research_old.oms import *
from gnomepy.research_old.trade_signal import TradeSignal, BasketTradeSignal
import pandas as pd
import numpy as np
import datetime
from typing import List, Union

class Backtest:
    """A class for backtesting trading strategies using historical market data.

    This class handles fetching historical data, running strategies, and tracking orders/performance.

    Attributes:
        client (MarketDataClient): Client for fetching market data
        strategies (Strategy): Trading strategies to research_old
        start_datetime (datetime): Start time for research_old period
        end_datetime (datetime): End time for research_old period
        listing_data (dict): Historical market data for each listing
        signal_history (list): History of signals with their timestamps
    """

    def __init__(self, client: MarketDataClient, strategies: Strategy, start_datetime: datetime.datetime, end_datetime: datetime.datetime):
        """Initialize the research_old.

        Args:
            client (MarketDataClient): Client for fetching market data
            strategies (Strategy): Trading strategies to research_old
            start_datetime (datetime): Start time for research_old period
            end_datetime (datetime): End time for research_old period
        """
        self.client = client
        self.strategies = strategies
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.listing_data = self._fetch_data()
        self.signal_history = []  # List to store (timestamp, signal) tuples

    def _fetch_data(self) -> pd.DataFrame:
        """Fetch and align historical market data for all listings used by the strategies.
        
        Returns:
            dict: Dictionary mapping Listing objects to their historical data DataFrames
        """
        listing_data = {}
        self.max_ticks = 0
        reference_timestamps = None

        # Get unique listings across all strategies
        for strategy in self.strategies:
            for listing in strategy.listings:
                if listing not in listing_data:
                    client_data_params = {
                        "exchange_id": listing.exchange_id,
                        "security_id": listing.security_id,
                        "start_datetime": self.start_datetime,
                    
                        "end_datetime": self.end_datetime,
                        "schema_type": strategy.data_schema_type,
                    }
                    current_listing_data = self.client.get_data(**client_data_params)
                    df = current_listing_data.to_df()
                    
                    # Use first listing's timestamps as reference
                    if reference_timestamps is None:
                        reference_timestamps = df['timestampEvent'].values
                        self.max_ticks = len(df)
                        listing_data[listing] = df
                    else:
                        # For subsequent listings, align to closest timestamp
                        aligned_indices = np.searchsorted(df['timestampEvent'].values, reference_timestamps)
                        # Clip to ensure we don't go out of bounds
                        aligned_indices = np.clip(aligned_indices, 0, len(df) - 1)

                        listing_data[listing] = df.iloc[aligned_indices].reset_index(drop=True)

        return listing_data

    def compute_portfolio_metrics(self, order_log) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute portfolio metrics using order history and price data.
        
        Args:
            order_log: List of dictionaries containing order information
            
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Portfolio metrics and trade history DataFrames
        """
        metrics = pd.DataFrame()
        history = pd.DataFrame()
        
        return metrics, history

    def run(self, data_type: str = 'pandas') -> List[Union[pd.DataFrame, np.ndarray]]:
        """Run the research_old simulation.

        Processes historical data through the strategies and order management system,
        tracking orders and performance metrics.

        Args:
            data_type (str, optional): Format of output data. Defaults to 'pandas'.

        Returns:
            List[Union[pd.DataFrame, np.ndarray]]: Portfolio performance metrics
        """
        # First initialize the Strategy for backtesting
        for strategy in self.strategies:
            strategy.initialize_backtest()

        # Then initalize OMS
        oms = OMS(strategies=self.strategies, notional=100, starting_cash=1e5)
        order_log = []  # List of {strategy: order} dictionaries
        
        # Iterate through each timestamp in the dataset with progress bar
        from tqdm import tqdm
        for idx in tqdm(range(0, self.max_ticks), desc="Processing ticks", unit="tick"):
            # Initialize list to collect all signals
            all_signals = []
            
            # Iterate through each strategy
            for strategy in self.strategies:
                # Get updated idx
                sampled_idx = idx // strategy.trade_frequency

                # We need enough data to complete strategy. We also only want to execute the trade at the correct frequency
                if sampled_idx >= strategy.max_lookback and idx % strategy.trade_frequency == 0:
                    strategy_data = {}
                    for listing in strategy.listings:
                        strategy_data[listing] = self.listing_data[listing].iloc[::strategy.trade_frequency].reset_index(drop=True).loc[sampled_idx - strategy.max_lookback:sampled_idx]

                    # Process new event
                    signals, latency = strategy.process_event(listing_data=strategy_data)

                    # Add signals to list if there are any
                    if signals and len(signals) > 0:
                        current_timestamp = self.listing_data[list(strategy_data.keys())[0]].iloc[sampled_idx]['timestampEvent']
                        for signal in signals:
                            all_signals.append(signal)
                            self.signal_history.append((current_timestamp, signal))
                else:
                    continue

            # Send all collected signals to OMS
            if all_signals and len(all_signals) > 0:
                filled_orders = oms.process_signals(signals=all_signals, lisings_lob_data=strategy_data)
                if filled_orders:
                    order_log.extend(filled_orders)  # Extend with list of {strategy: order} dicts

        return self.compute_portfolio_metrics(order_log)


class VectorizedBacktest(Backtest):
    """A vectorized version of the research_old that precomputes strategy values for faster execution.
    
    This class reuses most of the existing research_old infrastructure but precomputes
    strategy-dependent values (like beta vectors for cointegration strategies) to
    avoid recalculating them at each tick.
    """

    def __init__(self, client: MarketDataClient, strategies: Strategy, start_datetime: datetime.datetime, end_datetime: datetime.datetime):
        """Initialize the vectorized research_old.
        
        Args:
            client (MarketDataClient): Client for fetching market data
            strategies (Strategy): Trading strategies to research_old
            start_datetime (datetime): Start time for research_old period
            end_datetime (datetime): End time for research_old period
        """
        super().__init__(client, strategies, start_datetime, end_datetime)
        self.precomputed_values = {}
        # Store sampled data to avoid resampling
        self.sampled_listing_data = {}

    def _precompute_strategy_values(self):
        """Precompute all strategy-dependent values for vectorized execution."""
        for strategy in self.strategies:
            strategy_hash = str(strategy)
            
            # Sample data at trade frequency intervals if not already sampled
            if not self.sampled_listing_data:
                for listing in strategy.listings:
                    sampled_data = self.listing_data[listing].iloc[::strategy.trade_frequency].reset_index(drop=True)
                    self.sampled_listing_data[listing] = sampled_data
            
            self.precomputed_values[strategy_hash] = strategy.precompute_strategy_values(self.sampled_listing_data)

    def run(self, data_type: str = 'pandas') -> List[Union[pd.DataFrame, np.ndarray]]:
        """Run the vectorized research_old simulation.
        
        This method precomputes strategy values and generates all signals at once,
        then processes them through the OMS in chronological order.

        Args:
            data_type (str, optional): Format of output data. Defaults to 'pandas'.

        Returns:
            List[Union[pd.DataFrame, np.ndarray]]: Portfolio performance metrics
        """
        # Initialize strategies
        for strategy in self.strategies:
            strategy.initialize_backtest()

        # Precompute strategy values
        self._precompute_strategy_values()

        # Initialize OMS
        oms = OMS(strategies=self.strategies, notional=100, starting_cash=1e5)
        order_log = []  # List of {strategy: order} dictionaries

        # Generate all signals for all strategies
        all_signals = []
        for strategy in self.strategies:
            strategy_hash = str(strategy)
            precomputed_values = self.precomputed_values[strategy_hash]
            signals = strategy.generate_vectorized_signals(self.sampled_listing_data, precomputed_values)
            all_signals.extend(signals)
            # Store signals in history
            self.signal_history.extend(signals)  # signals are already (timestamp, signal) tuples

        # Sort signals by timestamp for chronological processing
        # Signals are already returned as (timestamp, signal) tuples
        all_signals.sort(key=lambda x: x[0])  # Sort by timestamp

        # Process signals chronologically
        from tqdm import tqdm
        for timestamp, signal in tqdm(all_signals, desc="Processing signals", unit="signal"):
            # Get current market data for this timestamp
            current_listing_data = {}
            
            # Determine which listings we need data for
            if isinstance(signal, BasketTradeSignal):
                listings_needed = [s.listing for s in signal.signals]
                strategy = signal.strategy
            else:
                listings_needed = [signal.listing]
                strategy = None
            
            for listing in listings_needed:
                if listing in self.listing_data:
                    # Find the closest timestamp in the original (unsampled) data
                    timestamps = self.listing_data[listing]['timestampEvent'].values
                    idx = np.searchsorted(timestamps, timestamp)
                    if idx >= len(timestamps):
                        idx = len(timestamps) - 1
                    
                    # Get a window of data around this timestamp for the strategy
                    if strategy:
                        # Use the same window logic as iterative research_old
                        sampled_idx = idx // strategy.trade_frequency
                        if sampled_idx >= strategy.max_lookback:
                            window_start = max(0, sampled_idx - strategy.max_lookback)
                            window_end = sampled_idx + 1
                            # Use already sampled data
                            current_listing_data[listing] = self.sampled_listing_data[listing].iloc[window_start:window_end]
                        else:
                            # Not enough data, skip this signal
                            continue
                    else:
                        # For single signals, just get the current data point
                        current_listing_data[listing] = self.listing_data[listing].iloc[idx:idx+1]

            # Process the signal through OMS
            filled_orders = oms.process_signals(signals=[signal], lisings_lob_data=current_listing_data)
            if filled_orders:
                order_log.extend(filled_orders)

        return self.compute_portfolio_metrics(order_log)