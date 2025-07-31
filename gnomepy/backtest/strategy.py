from abc import ABC, abstractmethod

from gnomepy.data.types import Order, OrderExecutionReport
from gnomepy.backtest.latency import LatencyModel
from gnomepy.data.common import DataStore

class Strategy(ABC):

    def __init__(self, processing_latency: LatencyModel):
        self.processing_latency = processing_latency

    @abstractmethod
    def on_market_data(self, data: DataStore) -> list[Order]:
        ...

    @abstractmethod
    def on_execution_report(self, execution_report: OrderExecutionReport):
        ...

    def simulate_strategy_processing_time(self) -> int:
        return self.processing_latency.simulate()
