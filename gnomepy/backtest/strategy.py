from abc import ABC, abstractmethod

from gnomepy.data.types import Order, OrderExecutionReport
from gnomepy.backtest.latency import LatencyModel
from gnomepy.data.common import DataStore
from gnomepy.backtest.recorder import Recorder


class Strategy(ABC):

    def __init__(self, processing_latency: LatencyModel):
        self.processing_latency = processing_latency

    @abstractmethod
    def on_market_data(self, timestamp: int, data: DataStore, recorder: Recorder) -> list[Order]:
        ...

    @abstractmethod
    def on_execution_report(self, timestamp: int, execution_report: OrderExecutionReport, recorder: Recorder):
        ...

    def simulate_strategy_processing_time(self) -> int:
        return self.processing_latency.simulate()
