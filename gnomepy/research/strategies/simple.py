from gnomepy import Strategy, LatencyModel
from gnomepy.research.oms import BaseOMS


class SimpleStrategy(Strategy):
    def __init__(self, processing_latency: LatencyModel, oms: BaseOMS):
        super().__init__(processing_latency)
        self.oms = oms

    def on_market_data(self, timestamp, data, recorder=None):
        return self.oms.on_market_update(timestamp, data, recorder)

    def on_execution_report(self, timestamp, execution_report, recorder=None):
        self.oms.on_execution_report(timestamp, execution_report, recorder)
