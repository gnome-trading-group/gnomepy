from __future__ import annotations

from abc import ABC, abstractmethod

from gnomepy.java.backtest.orders import ExecutionReport
from gnomepy.java.oms import Intent
from gnomepy.java.schemas import Schema


class Strategy(ABC):
    """Base class for Python backtest strategies.

    Strategies declare desired state via Intents. The OMS resolves intents
    into orders (diffing against current open orders) and handles risk
    validation, order tracking, and position management.

    Timestamps are available on the data objects themselves:
        data.event_timestamp  — for market data
        report.timestamp_recv — for execution reports

    Example:
        class MyMMStrategy(Strategy):
            def on_market_data(self, data):
                mid = (data.bid_price(0) + data.ask_price(0)) // 2
                return [Intent(
                    exchange_id=1, security_id=100,
                    bid_price=mid - 50, bid_size=10,
                    ask_price=mid + 50, ask_size=10,
                )]

            def on_execution_report(self, report):
                print(f"Fill: {report.fill_price} x {report.filled_qty}")
                return []
    """

    @abstractmethod
    def on_market_data(self, data: Schema) -> list[Intent]:
        """Called on each market data update. Return desired state as Intents."""
        ...

    @abstractmethod
    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        """Called when an execution report is received. Return new Intents if desired."""
        ...

    _oms_view = None

    @property
    def oms(self):
        """Access the OMS for position/order queries. Available after backtest starts."""
        return self._oms_view

    def simulate_processing_time(self) -> int:
        """Override to simulate strategy processing latency in nanoseconds."""
        return 0
