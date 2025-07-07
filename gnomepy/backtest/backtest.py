from gnomepy.data.client import MarketDataClient
from gnomepy.data.types import SchemaType
from gnomepy.backtest.archive.strategy_old import Strategy
from gnomepy.backtest.strategy import *
from gnomepy.backtest.oms import *
import pandas as pd
import numpy as np
import datetime
from typing import List, Union

class Backtest:
    """A class for backtesting trading strategies using historical market data.

    This class handles fetching historical data, running strategies, and tracking orders/performance.

    Attributes:
        client (MarketDataClient): Client for fetching market data
        strategies (Strategy): Trading strategies to backtest
        start_datetime (datetime): Start time for backtest period
        end_datetime (datetime): End time for backtest period
        listing_data (dict): Historical market data for each listing
    """

    def __init__(self, client: MarketDataClient, strategies: Strategy, start_datetime: datetime.datetime, end_datetime: datetime.datetime):
        """Initialize the backtest.

        Args:
            client (MarketDataClient): Client for fetching market data
            strategies (Strategy): Trading strategies to backtest
            start_datetime (datetime): Start time for backtest period
            end_datetime (datetime): End time for backtest period
        """
        self.client = client
        self.strategies = strategies
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.listing_data = self._fetch_data()

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
        Calculates metrics like P&L, profit factor, drawdown, win rate etc.
        
        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            DataFrame containing portfolio metrics and trade history DataFrame
        """
        # Create history dataframe from order log
        history = pd.DataFrame([{
            'timestamp': order.timestampOpened,
            'listing': order.listing,
            'action': order.action.value,
            'size': order.size,
            'price': order.price,
            'cash_flow': order.cash_size,
            'strategy': strategy_hash,
            'signal_type': order.signal.signal_type if isinstance(order.signal, BasketSignal) else 'single'
        } for order_dict in order_log 
          for strategy_hash, order in order_dict.items()])
        
        if len(history) == 0:
            return pd.DataFrame(), pd.DataFrame()
            
        # Sort by timestamp
        history = history.sort_values('timestamp')
        
        # Calculate P&L per trade
        history['pl'] = history['cash_flow']
        history['cum_pl'] = history.groupby('strategy')['pl'].cumsum()
        
        # Calculate metrics per strategy
        metrics_list = []
        for strategy_name in history['strategy'].unique():
            strategy_history = history[history['strategy'] == strategy_name]
            
            total_trades = len(strategy_history)
            winning_trades = len(strategy_history[strategy_history['pl'] > 0])
            losing_trades = len(strategy_history[strategy_history['pl'] < 0])
            
            metrics = {
                'strategy': strategy_name,
                'total_pl': strategy_history['pl'].sum(),
                'profit_factor': abs(strategy_history[strategy_history['pl'] > 0]['pl'].sum()) / abs(strategy_history[strategy_history['pl'] < 0]['pl'].sum()) if abs(strategy_history[strategy_history['pl'] < 0]['pl'].sum()) != 0 else float('inf'),
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'avg_pl_per_trade': strategy_history['pl'].mean(),
                'std_pl_per_trade': strategy_history['pl'].std(),
                'max_drawdown': (strategy_history['cum_pl'] - strategy_history['cum_pl'].expanding().max()).min(),
                'sharpe_ratio': strategy_history['pl'].mean() / strategy_history['pl'].std() * np.sqrt(252) if strategy_history['pl'].std() != 0 else 0,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades
            }
            metrics_list.append(metrics)
        
        return pd.DataFrame(metrics_list), history

    def run(self, data_type: str = 'pandas') -> List[Union[pd.DataFrame, np.ndarray]]:
        """Run the backtest simulation.

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
                        all_signals.extend(signals)
                else:
                    continue

            # Send all collected signals to OMS
            if all_signals and len(all_signals) > 0:
                filled_orders = oms.process_signals(signals=all_signals, lisings_lob_data=strategy_data)
                if filled_orders:
                    order_log.extend(filled_orders)  # Extend with list of {strategy: order} dicts

        return self.compute_portfolio_metrics(order_log)