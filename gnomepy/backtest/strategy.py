from abc import ABC, abstractmethod

from gnomepy import SchemaBase, Order, OrderExecutionReport, LatencyModel


class Strategy(ABC):

    def __init__(self, processing_latency: LatencyModel):
        self.processing_latency = processing_latency

    @abstractmethod
    def on_market_data(self, data: SchemaBase) -> list[Order]:
        raise NotImplementedError

    @abstractmethod
    def on_execution_report(self, execution_report: OrderExecutionReport):
        raise NotImplementedError

    def simulate_strategy_processing_time(self) -> int:
        return self.processing_latency.simulate()
