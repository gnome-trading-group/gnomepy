from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import timedelta

from gnomepy.java.backtest.orders import ExecutionReport
from gnomepy.java.oms import Intent
from gnomepy.java.schemas import StreamRecord
from gnomepy.java.streams import StreamDirectory

logger = logging.getLogger(__name__)


class Strategy(ABC):
    """Base class for Python backtest strategies.

    Strategies declare desired state via Intents. The OMS resolves intents
    into orders (diffing against current open orders) and handles risk
    validation, order tracking, and position management.

    Timestamps are available on the data objects themselves:
        data.event_timestamp  — for market data
        report.timestamp_recv — for execution reports

    Identity:
        Market data names only its stream (``data.stream_id``); orders, intents and
        execution reports name a listing by exchange and security. ``self.streams``
        resolves both to a listing id, so key state by listing:
        ``self.streams.listing_id(data.stream_id)`` and
        ``self.streams.listing_id_for(report.exchange_id, report.security_id)``.

    Custom metrics:
        Override ``register_metrics`` to declare named metric streams: add
        typed columns, call ``freeze()``, then record each row with
        ``stream.begin()``, ``stream.set_double(col, val)`` per column and
        ``stream.commit()``. A backtest keeps them in its results; a live or
        paper session journals them (``gnomepy.journal.load_session_metrics``).

    Example:
        class MyMMStrategy(Strategy):
            def register_metrics(self):
                stream = self.metrics.create_stream("signals")
                self.ts_col = stream.add_long_column("timestamp")
                self.fv_col = stream.add_double_column("fair_value")
                stream.freeze()
                self.signals = stream

            def on_market_data(self, data):
                listing = self.streams.listing(self.streams.listing_id(data.stream_id))
                mid = (data.bid_price(0) + data.ask_price(0)) // 2
                self.signals.begin()
                self.signals.set_long(self.ts_col, data.event_timestamp)
                self.signals.set_double(self.fv_col, mid / 1e9)
                self.signals.commit()
                return [Intent(
                    exchange_id=listing.exchange().exchangeId(),
                    security_id=listing.security().securityId(),
                    bid_price=mid - 50, bid_size=10,
                    ask_price=mid + 50, ask_size=10,
                )]

            def on_execution_report(self, report):
                print(f"Fill: {report.fill_price} x {report.filled_qty}")
                return []
    """

    @abstractmethod
    def on_market_data(self, data: StreamRecord) -> list[Intent]:
        """Called on each book record (a venue listing's market data). Return desired state as Intents."""
        ...

    def on_stream_event(self, stream_id: int, record: StreamRecord) -> list[Intent]:
        """Called on each record of any other stream the strategy reads (a vendor feed, say). None by default."""
        return []

    @abstractmethod
    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        """Called when an execution report is received. Return new Intents if desired."""
        ...

    _position_view = None
    _metric_recorder = None
    _control = None
    _streams = None

    @property
    def streams(self) -> StreamDirectory:
        """The streams this strategy reads and the listings they carry. Available before any data arrives."""
        return self._streams

    @property
    def positions(self):
        """Access position data. Available after the strategy is initialized."""
        return self._position_view

    @property
    def metrics(self):
        """Access the MetricRecorder to create custom metric streams.

        Available inside ``register_metrics()`` and throughout the run.
        """
        return self._metric_recorder

    def request_exit(self, reason: str, *, flatten: bool = False, timeout: timedelta = timedelta(minutes=5)) -> bool:
        """End this session, e.g. a pregame strategy once its match has started. Safe to call from any thread.

        From then on ``on_market_data`` and ``on_execution_report`` are no longer called. The session cancels the
        strategy's quotes and, with ``flatten``, closes every position with market orders until it is flat or
        ``timeout`` has passed. It then raises a SESSION_EXITED alert (a WARNING if positions it was asked to close
        are still held), and stops: it shows as STOPPED with ``reason``. Without ``flatten``, positions are kept,
        e.g. to hold them to settlement.

        Only the first request counts. In a backtest the strategy just goes quiet.

        Returns:
            True if this request is the one the session acts on.
        """
        if self._control is None:
            logger.warning("request_exit(%r) before the strategy was initialized; ignored", reason)
            return False
        return bool(self._control.requestExit(reason, flatten, int(timeout.total_seconds() * 1000)))

    @property
    def is_exiting(self) -> bool:
        """True once an exit has been requested."""
        return self._control is not None and bool(self._control.isExiting())

    def register_metrics(self) -> None:
        """Override to declare custom metric streams before any data arrives.

        Called once ``self.metrics`` and ``self.positions`` are set, in a
        backtest and in a live or paper session alike. Create streams, add
        columns, and freeze them here.
        """

    def simulate_processing_time(self) -> int:
        """Simulated time in nanoseconds between seeing an event and orders leaving.

        The strategy handles no other event meanwhile: market data and execution
        reports arriving during it wait their turn, in arrival order. 0 (the
        default) means none. Ignored when the backtest config sets
        ``measure_processing_time``, which charges measured wall-clock time instead.
        """
        return 0
