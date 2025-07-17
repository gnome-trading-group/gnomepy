from gnomepy import SimulatedExchange, SchemaType, Order, OrderExecutionReport, FeeModel, LatencyModel
from gnomepy.backtest.queues import QueueModel


class MBPSimulatedExchange(SimulatedExchange):

    def __init__(
            self,
            fee_model: FeeModel,
            network_latency: LatencyModel,
            order_processing_latency: LatencyModel,
            queue_model: QueueModel,
    ):
        super().__init__(fee_model, network_latency, order_processing_latency)

    def submit_order(self, order: Order) -> OrderExecutionReport:
        # TODO: IMPLEMENT THIS!!!!!!!!!!!!!!!!!!!!!!!
        pass

    def get_supported_schemas(self) -> list[SchemaType]:
        return [SchemaType.MBP_10, SchemaType.MBP_1, SchemaType.BBO_1M, SchemaType.BBO_1S]
