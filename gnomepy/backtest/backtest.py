from gnomepy.data.client import MarketDataClient
from gnomepy.data.types import SchemaType
from gnomepy.backtest.archive.strategy_old import Strategy
from gnomepy.backtest.strategy import *
from gnomepy.backtest.oms import *
import pandas as pd
import numpy as np
import datetime
from typing import List, Union

# How can we backtest multiple strategies at once?
# 1. If they all have the same trading frequency then it is easy as we can step through each time step with ease.
# 2. If there's different frequency then we have to do something different. Maybe we iterate through each data point unsampled, then dynamically create a sampled
# index which is just a division.
class Backtest:
    def __init__(self, client: MarketDataClient, strategies: Strategy, start_datetime: datetime.datetime, end_datetime: datetime.datetime):
        self.client = client
        self.strategies = strategies
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.listing_data = self._fetch_data()

    def _fetch_data(self) -> pd.DataFrame:
        listing_data = {}
        self.max_ticks = 0

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
                    
                    # Set max_ticks
                    self.max_ticks = len(current_listing_data.to_df())
                    
                    # Convert to dataframe without resampling
                    listing_data[listing] = current_listing_data.to_df()

        return listing_data

    def run(self, data_type: str = 'pandas') -> List[Union[pd.DataFrame, np.ndarray]]:

        print(f"Starting backtest from {self.start_datetime} to {self.end_datetime}")
        print(f"Total ticks to process: {self.max_ticks}")

        # First initialize the Strategy for backtesting
        for strategy in self.strategies:
            strategy.initialize_backtest()
            print(f"Initialized strategy: {strategy.__class__.__name__}")

        # Then initalize OMS
        oms = OMS(strategies=self.strategies, notional=100, starting_cash=1e5)
        order_log = []  # List of {strategy: order} dictionaries

        # Iterate through eaach timestamp in the dataset
        for idx in range(0, self.max_ticks):

            # Initialize list to collect all signals
            all_signals = []
            
            # Iterate through each strategy
            for strategy in self.strategies:

                # Get updated idx
                sampled_idx = idx // strategy.trade_frequency

                # We need enough data to complete strategy. We also only want to execute the trade at the correct frequency
                if sampled_idx >= strategy.max_lookback and idx % strategy.trade_frequency == 0:
                    print(f"Processing tick {idx}")
                    print(f"Processing tick {sampled_idx}")
                    # We need to make sure the data looks correct for all strategies
                    strategy_data = {}
                    for listing in strategy.listings:
                        strategy_data[listing] = self.listing_data[listing].iloc[::strategy.trade_frequency].reset_index(drop=True).loc[sampled_idx - strategy.max_lookback:sampled_idx]

                    # Process new event
                    signals, latency = strategy.process_event(listing_data=strategy_data)

                    # Add signals to list if there are any
                    if signals and len(signals) > 0:
                        print(f"Generated {len(signals)} signals from {strategy.__class__.__name__} at tick {idx}")
                        all_signals.extend(signals)
                else:
                    continue

            # Send all collected signals to OMS
            if all_signals and len(all_signals) > 0:
                print(f"Processing {len(all_signals)} total signals at tick {idx}:")
                for signal in all_signals:
                    print(f"  Signal: {signal}")
                filled_orders = oms.process_signals(signals=all_signals, lisings_lob_data=strategy_data)
                if filled_orders:
                    print(f"Generated {len(filled_orders)} filled orders")
                    order_log.extend(filled_orders)  # Extend with list of {strategy: order} dicts
        
        print(f"Backtest complete. Generated {len(order_log)} order logs")
        return oms.compute_portfolio_metrics(self.listing_data)