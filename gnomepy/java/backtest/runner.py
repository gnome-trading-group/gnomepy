from __future__ import annotations

from datetime import date, datetime

import jpype
from jpype import JImplements, JOverride

from gnomepy.java._jvm import ensure_jvm_started
from gnomepy.java.enums import SchemaType
from gnomepy.java.schemas import wrap_schema
from gnomepy.java.backtest.orders import ExecutionReport
from gnomepy.java.backtest.strategy import Strategy
from gnomepy.java.backtest.config import ExchangeConfig
from gnomepy.java.oms import Intent


def _create_java_strategy(py_strategy: Strategy, adapter):
    """Create a JPype proxy that implements the Java BacktestStrategy interface.

    Args:
        py_strategy: Python Strategy implementation.
        adapter: Java OmsBacktestAdapter instance.
    """

    ArrayList = jpype.JClass("java.util.ArrayList")

    _java_intent = None

    def _ensure_intent_class():
        nonlocal _java_intent
        if _java_intent is None:
            _java_intent = jpype.JClass("group.gnometrading.oms.intent.Intent")

    def _to_java_intent(intent):
        _ensure_intent_class()
        ji = _java_intent()
        has_take = intent.take_size > 0 and intent.take_side is not None
        has_quote = intent.bid_size > 0 or intent.ask_size > 0
        if has_quote and has_take:
            ji.setQuoteAndTake(
                int(intent.exchange_id), jpype.JLong(intent.security_id),
                int(intent.strategy_id),
                jpype.JLong(intent.bid_price), jpype.JLong(intent.bid_size),
                jpype.JLong(intent.ask_price), jpype.JLong(intent.ask_size),
                intent.take_side.to_java(), jpype.JLong(intent.take_size),
                intent.take_order_type.to_java(), jpype.JLong(intent.take_limit_price),
            )
        elif has_take:
            ji.setTake(
                int(intent.exchange_id), jpype.JLong(intent.security_id),
                int(intent.strategy_id),
                intent.take_side.to_java(), jpype.JLong(intent.take_size),
                intent.take_order_type.to_java(), jpype.JLong(intent.take_limit_price),
            )
        else:
            ji.setQuote(
                int(intent.exchange_id), jpype.JLong(intent.security_id),
                int(intent.strategy_id),
                jpype.JLong(intent.bid_price), jpype.JLong(intent.bid_size),
                jpype.JLong(intent.ask_price), jpype.JLong(intent.ask_size),
            )
        return ji

    @JImplements("group.gnometrading.backtest.driver.BacktestStrategy")
    class _Proxy:
        @JOverride
        def onMarketData(self, timestamp, data):
            wrapped = wrap_schema(data)
            intents = py_strategy.on_market_data(int(timestamp), wrapped)
            if not intents:
                return ArrayList()

            # Convert Python intents to Java array
            _ensure_intent_class()
            intent_array = jpype.JArray(_java_intent)(len(intents))
            for i, py_intent in enumerate(intents):
                intent_array[i] = _to_java_intent(py_intent)

            # Java handles everything: resolve → validate → track → LocalMessage
            return adapter.processIntents(timestamp, intent_array, len(intents))

        @JOverride
        def onExecutionReport(self, timestamp, report):
            messages = adapter.processExecutionReport(report)
            py_report = ExecutionReport._from_java(report)
            py_strategy.on_execution_report(int(timestamp), py_report)
            return messages

        @JOverride
        def simulateProcessingTime(self):
            return jpype.JLong(py_strategy.simulate_processing_time())

    return _Proxy()


def _build_exchange_map(configs: list[ExchangeConfig]):
    """Build Map<Integer, Map<Integer, SimulatedExchange>> from Python configs."""
    HashMap = jpype.JClass("java.util.HashMap")
    MbpSimulatedExchange = jpype.JClass(
        "group.gnometrading.backtest.exchange.MbpSimulatedExchange"
    )

    outer = HashMap()
    for cfg in configs:
        exchange = MbpSimulatedExchange(
            cfg.fee_model._to_java(),
            cfg.network_latency._to_java(),
            cfg.order_processing_latency._to_java(),
            cfg.queue_model._to_java(),
        )

        java_exchange_id = jpype.JInt(cfg.exchange_id)
        java_security_id = jpype.JInt(cfg.security_id)
        inner = outer.get(java_exchange_id)
        if inner is None:
            inner = HashMap()
            outer.put(java_exchange_id, inner)
        inner.put(java_security_id, exchange)

    return outer


class Backtest:
    """Orchestrate a backtest with a Python strategy against Java simulation.

    Usage:
        backtest = Backtest(
            strategy=MyStrategy(),
            schema_type=SchemaType.MBP_10,
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 16),
            exchanges=[
                ExchangeConfig(exchange_id=1, security_id=100),
            ],
        )
        backtest.run()
    """

    def __init__(
        self,
        strategy: Strategy,
        schema_type: SchemaType,
        start_date: date | datetime,
        end_date: date | datetime,
        exchanges: list[ExchangeConfig],
        entries: list | None = None,
        bucket: str = "gnome-market-data-prod",
        s3_client=None,
        record: bool = True,
        risk_config=None,
    ):
        ensure_jvm_started()

        self._strategy = strategy
        self._schema_type = schema_type
        self._start_date = start_date
        self._end_date = end_date
        self._exchanges = exchanges
        self._entries = entries
        self._bucket = bucket
        self._s3_client = s3_client
        self._record = record
        self._risk_config = risk_config
        self._recorder = None
        self._driver = None

    def _build_driver(self):
        LocalDateTime = jpype.JClass("java.time.LocalDateTime")

        # Convert to datetime if date was passed
        start = self._start_date if isinstance(self._start_date, datetime) else datetime(
            self._start_date.year, self._start_date.month, self._start_date.day
        )
        end = self._end_date if isinstance(self._end_date, datetime) else datetime(
            self._end_date.year, self._end_date.month, self._end_date.day
        )

        java_start = LocalDateTime.of(start.year, start.month, start.day, start.hour, start.minute, start.second)
        java_end = LocalDateTime.of(end.year, end.month, end.day, end.hour, end.minute, end.second)
        java_schema_type = self._schema_type.to_java()

        # Build entries if not provided
        if self._entries is None:
            from gnomepy.java.market_data import MarketDataClient

            client = MarketDataClient(bucket=self._bucket, s3_client=self._s3_client)
            entries = []
            for cfg in self._exchanges:
                entries.extend(
                    client.get_java_entries(
                        cfg.security_id, cfg.exchange_id,
                        self._schema_type, start, end,
                    )
                )
            ArrayList = jpype.JClass("java.util.ArrayList")
            java_entries = ArrayList()
            for e in entries:
                java_entries.add(e)
        else:
            ArrayList = jpype.JClass("java.util.ArrayList")
            java_entries = ArrayList()
            for e in self._entries:
                java_entries.add(e)

        # Build exchange map
        exchange_map = _build_exchange_map(self._exchanges)

        # Create recorder (before adapter so it can be passed in)
        if self._record:
            JavaRecorder = jpype.JClass(
                "group.gnometrading.backtest.recorder.BacktestRecorder"
            )
            self._recorder = JavaRecorder()

        # Always create OMS + backtest adapter
        from gnomepy.java.oms import _build_java_oms, OmsView, RiskConfig

        risk_config = self._risk_config if self._risk_config is not None else RiskConfig()
        java_oms = _build_java_oms(risk_config)
        self._strategy._oms_view = OmsView(java_oms)

        OmsBacktestAdapter = jpype.JClass(
            "group.gnometrading.backtest.oms.OmsBacktestAdapter"
        )
        adapter = OmsBacktestAdapter(java_oms, self._recorder) if self._recorder else OmsBacktestAdapter(java_oms)

        # Create Java strategy proxy
        java_strategy = _create_java_strategy(self._strategy, adapter)

        # Create S3 client
        if self._s3_client is None:
            S3Client = jpype.JClass("software.amazon.awssdk.services.s3.S3Client")
            s3 = S3Client.create()
        else:
            s3 = self._s3_client

        # Build the driver
        BacktestDriver = jpype.JClass(
            "group.gnometrading.backtest.driver.BacktestDriver"
        )
        self._driver = BacktestDriver(
            java_start,
            java_end,
            java_entries,
            java_schema_type,
            java_strategy,
            exchange_map,
            s3,
            str(self._bucket),
        )

        # Attach recorder to driver (for market data recording)
        if self._recorder is not None:
            self._driver.setRecorder(self._recorder)

    def run(self, progress: bool = True) -> BacktestResults | None:
        """Prepare data and fully execute the backtest.

        Args:
            progress: If True, print progress updates during execution.
        """
        from gnomepy.java.recorder import BacktestResults

        self._build_driver()
        self._driver.prepareData()

        if not progress:
            self._driver.fullyExecute()
        else:
            self._run_with_progress()

        if self._recorder is not None:
            return BacktestResults(self._recorder)
        return None

    def _run_with_progress(self):
        """Execute backtest with progress reporting."""
        import sys
        import time

        start = self._start_date
        end = self._end_date
        total_sec = (end - start).total_seconds()
        if total_sec <= 0:
            self._driver.fullyExecute()
            return

        # Convert to nanosecond timestamps (UTC)
        import pytz
        epoch = datetime(1970, 1, 1, tzinfo=pytz.UTC)
        start_ns = int((start.replace(tzinfo=pytz.UTC) - epoch).total_seconds() * 1_000_000_000)
        end_ns = int((end.replace(tzinfo=pytz.UTC) - epoch).total_seconds() * 1_000_000_000)

        # Execute in 1-minute sim-time chunks
        chunk_ns = 60_000_000_000
        current_ns = start_ns
        t0 = time.time()

        while current_ns < end_ns:
            current_ns += chunk_ns
            self._driver.executeUntil(jpype.JLong(min(current_ns, end_ns)))

            elapsed = time.time() - t0
            pct = min((current_ns - start_ns) / (end_ns - start_ns) * 100, 100)
            events = int(self._driver.getEventsProcessed())
            print(f"\rBacktest: {pct:.0f}% | events: {events:,} | {elapsed:.1f}s", end="")
            sys.stdout.flush()

        # Process any remaining events beyond end timestamp
        self._driver.fullyExecute()
        elapsed = time.time() - t0
        events = int(self._driver.getEventsProcessed())
        print(f"\rBacktest: 100% | events: {events:,} | {elapsed:.1f}s")
        sys.stdout.flush()

    def run_until(self, timestamp: int) -> BacktestResults | None:
        """Run the backtest until a specific timestamp."""
        from gnomepy.java.recorder import BacktestResults

        if self._driver is None:
            self._build_driver()
            self._driver.prepareData()
        self._driver.executeUntil(jpype.JLong(timestamp))

        if self._recorder is not None:
            return BacktestResults(self._recorder)
        return None
