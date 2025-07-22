from gnomepy import Strategy, LatencyModel, SchemaBase, Order, OrderExecutionReport
from gnomepy.research.oms import SimpleOMS


class CointegrationOMSStrategy(Strategy):

    def __init__(self, processing_latency: LatencyModel, oms: SimpleOMS):
        super().__init__(processing_latency)
        self.oms = oms

    def on_market_data(self, data: SchemaBase) -> list[Order]:
        return self.oms.on_market_update(data)

    def on_execution_report(self, execution_report: OrderExecutionReport):
        self.oms.on_execution_report(execution_report)
