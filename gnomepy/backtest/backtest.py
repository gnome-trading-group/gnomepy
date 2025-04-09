from gnomepy.data.client import MarketDataClient
from gnomepy.data.types import SchemaType
from gnomepy.backtest.strategy import Strategy
import pandas as pd
import numpy as np
import datetime

class Backtest:
    def __init__(self, client: MarketDataClient, strategy: Strategy, exchange_id: int, security_id: int, start_datetime: datetime.datetime, end_datetime: datetime.datetime, schema_type: SchemaType):
        self.client = client
        self.strategy = strategy
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

    def run(self, data_type: str = 'pandas') -> pd.DataFrame | np.ndarray:
        if data_type == 'pandas':
            return self.strategy.execute(self.data.to_df())
        elif data_type == 'numpy':
            return self.strategy.execute(self.data.to_ndarray())
        else:
            raise ValueError("data_type must be either 'pandas' or 'numpy'")

    def get_strategy(self) -> Strategy:
        return self.strategy

    def get_client_data_params(self) -> dict:
        return self.client_data_params

    def get_data(self) -> pd.DataFrame:
        return self.data
