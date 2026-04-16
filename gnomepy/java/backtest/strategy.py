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

    Custom metrics:
        Override ``register_metrics`` to create named RecordBuffers. Each
        buffer is a columnar stream: add typed columns, call ``freeze()``,
        then write via ``buf.append_row()`` / ``buf.set_double(row, col, val)``.

    Example:
        class MyMMStrategy(Strategy):
            def register_metrics(self):
                buf = self.metrics.create_buffer("signals")
                self.ts_col = buf.add_long_column("timestamp")
                self.fv_col = buf.add_double_column("fair_value")
                buf.freeze()
                self.buf = buf

            def on_market_data(self, data):
                mid = (data.bid_price(0) + data.ask_price(0)) // 2
                row = self.buf.append_row()
                self.buf.set_long(row, self.ts_col, data.event_timestamp)
                self.buf.set_double(row, self.fv_col, mid / 1e9)
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
    _metric_recorder = None

    @property
    def oms(self):
        """Access the OMS for position/order queries. Available after backtest starts."""
        return self._oms_view

    @property
    def metrics(self):
        """Access the MetricRecorder to create custom record streams.

        Available inside ``register_metrics()`` and throughout the backtest.
        """
        return self._metric_recorder

    def register_metrics(self) -> None:
        """Override to declare custom metric streams before replay starts.

        Called by the framework after ``self.metrics`` is set but before data
        replay begins. Create buffers, add columns, and freeze them here.
        """

    def simulate_processing_time(self) -> int:
        """Override to simulate strategy processing latency in nanoseconds."""
        return 0
