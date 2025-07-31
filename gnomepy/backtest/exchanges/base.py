from abc import abstractmethod, ABC

from gnomepy.backtest.fee import FeeModel
from gnomepy.backtest.latency import LatencyModel
from gnomepy.data.types import SchemaType, Order, OrderExecutionReport, CancelOrder, SchemaBase


class SimulatedExchange(ABC):

    def __init__(
            self,
            fee_model: FeeModel,
            network_latency: LatencyModel,
            order_processing_latency: LatencyModel,
    ):
        self.fee_model = fee_model
        self.network_latency = network_latency
        self.order_processing_latency = order_processing_latency

    @abstractmethod
    def get_supported_schemas(self) -> list[SchemaType]:
        ...

    @abstractmethod
    def submit_order(self, order: Order) -> list[OrderExecutionReport]:
        ...

    @abstractmethod
    def cancel_order(self, order: CancelOrder) -> list[OrderExecutionReport]:
        ...

    @abstractmethod
    def on_market_data(self, data: SchemaBase) -> list[OrderExecutionReport]:
        ...

    def simulate_network_latency(self) -> int:
        # TODO: Do any exchanges face significantly different delivery vs receive latencies?
        return self.network_latency.simulate()

    def simulate_order_processing_time(self) -> int:
        return self.order_processing_latency.simulate()
