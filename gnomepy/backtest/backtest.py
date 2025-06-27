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
    def __init__(self, client: MarketDataClient, strategy: Strategy, start_datetime: datetime.datetime, end_datetime: datetime.datetime):
        self.client = client
        self.strategy = strategy
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.listing_data = self._fetch_data()

    def _fetch_data(self) -> pd.DataFrame:
        listing_data = {}

        for listing in self.strategy.listings:
            client_data_params = {
                "exchange_id": listing.exchange_id,
                "security_id": listing.security_id,
                "start_datetime": self.start_datetime,
                "end_datetime": self.end_datetime,
                "schema_type": self.strategy.data_schema_type,
            }
            current_listing_data = self.client.get_data(**client_data_params)

            # TODO: Confirm if this is the best place to do this, resampling now will help with data size. I think it's fine to have a list of dataframe or ndarrays personally, but you may disagree for schema sake @ MASON
            current_listing_df = current_listing_data.to_df()[::self.strategy.trade_frequency]
            listing_data[listing] = current_listing_df

        return listing_data

    def run(self, data_type: str = 'pandas') -> List[Union[pd.DataFrame, np.ndarray]]:

        # First initialize the Strategy for backtesting
        self.strategy.initialize_backtest()
        order_log = []

        # Then initalize OMS
        oms = OMS()

        # Iterate through eaach timestamp in the dataset
        for idx in range(0, len(self.listing_data[0])):

            # We need enough data to complete strategy.
            if idx > self.strategy.max_lookback:
                # Process new even
                signals, time_elapsed = self.strategy.process_event(listing_data=self.listing_data.iloc[idx - self.strategy.max_lookback:idx])

                # Send signals to OMS, if there are any to trade
                if signals and len(signals) > 0:
                    filled_orders = oms.process_signals(signals=signals)
                    order_log.append(filled_orders)

            else:
                continue

            