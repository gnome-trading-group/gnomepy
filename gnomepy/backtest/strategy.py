from abc import ABC, abstractmethod

from gnomepy.data.types import SchemaBase, Order, OrderExecutionReport
from gnomepy.backtest.latency import LatencyModel
from gnomepy.backtest.oms import SimpleOMS
from gnomepy.backtest.signal import Signal, CointegrationSignal
from gnomepy.data.common import DataStore

class Strategy(ABC):

    def __init__(self, processing_latency: LatencyModel):
        self.processing_latency = processing_latency

    @abstractmethod
    def on_market_data(self, data: DataStore) -> list[Order]:
        raise NotImplementedError

    @abstractmethod
    def on_execution_report(self, execution_report: OrderExecutionReport):
        raise NotImplementedError

    def simulate_strategy_processing_time(self) -> int:
        return self.processing_latency.simulate()


class CointegrationOMSStrategy(Strategy):

    def __init__(self, processing_latency: LatencyModel, oms: SimpleOMS):
        super().__init__(processing_latency)
        self.oms = oms

    def on_market_data(self, data: DataStore) -> list[Order]:       
        return self.oms.on_market_update(data)
    
    def on_execution_report(self, execution_report: OrderExecutionReport):
        self.oms.on_execution_report(execution_report)
    