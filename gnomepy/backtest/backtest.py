from gnomepy.data.client import MarketDataClient
from gnomepy.data.types import SchemaType
from gnomepy.backtest.strategy import Strategy
import pandas as pd
import numpy as np
import datetime
from typing import List, Union

class Backtest:
    def __init__(self, client: MarketDataClient, strategies: List[Strategy], exchange_id: int, security_id: int, start_datetime: datetime.datetime, end_datetime: datetime.datetime, schema_type: SchemaType):
        self.client = client
        self.strategies = strategies
        self.client_data_params = {
            "exchange_id": exchange_id,
            "security_id": security_id,
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            "schema_type": schema_type,
        }
        self.data = self._fetch_data()

    def _fetch_data(self) -> pd.DataFrame:
        return self.client.get_data(**self.client_data_params)

    def run(self, data_type: str = 'pandas') -> List[Union[pd.DataFrame, np.ndarray]]:
        results = []
        for strategy in self.strategies:
            if data_type == 'pandas':
                results.append(strategy.execute(self.data.to_df()))
            elif data_type == 'numpy':
                results.append(strategy.execute(self.data.to_ndarray()))
            else:
                raise ValueError("data_type must be either 'pandas' or 'numpy'")
        return results

    def get_strategies(self) -> List[Strategy]:
        return self.strategies

    def get_client_data_params(self) -> dict:
        return self.client_data_params

    def get_data(self) -> pd.DataFrame:
        return self.data
