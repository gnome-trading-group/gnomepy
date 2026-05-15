from __future__ import annotations

import importlib
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import jpype
import pytz
from jpype import JImplements, JOverride

from importlib.metadata import version as _pkg_version

logger = logging.getLogger(__name__)

from gnomepy.config import config as gnome_config
from gnomepy.java._jvm import ensure_jvm_started
from gnomepy.java.backtest.config import BacktestConfig
from gnomepy.java.backtest.orders import ExecutionReport
from gnomepy.java.backtest.strategy import Strategy
from gnomepy.java.oms import OmsView
from gnomepy.java.recorder import BacktestResults, MetricRecorder as PyMetricRecorder
from gnomepy.java.schemas import wrap_schema
from gnomepy.metadata import BacktestMetadata
from gnomepy.utils import generate_backtest_id


def _create_python_callback(py_strategy: Strategy):
    """Create a JPype proxy implementing PythonStrategyAgent.PythonStrategyCallback."""
    ArrayList = jpype.JClass("java.util.ArrayList")

    def _to_java_list(intents):
        lst = ArrayList()
        if intents:
            for intent in intents:
                lst.add(intent.raw)
        return lst

    @JImplements(
        "group.gnometrading.backtest.driver.PythonStrategyAgent$PythonStrategyCallback"
    )
    class _Proxy:
        @JOverride
        def onMarketData(self, data):
            wrapped = wrap_schema(data)
            intents = py_strategy.on_market_data(wrapped)
            return _to_java_list(intents)

        @JOverride
        def onExecutionReport(self, report):
            py_report = ExecutionReport._from_java(report)
            intents = py_strategy.on_execution_report(py_report)
            return _to_java_list(intents)

        @JOverride
        def simulateProcessingTime(self):
            return jpype.JLong(py_strategy.simulate_processing_time())

    return _Proxy()


def _instantiate_java_strategy(class_name: str, strategy_args: dict):
    """Resolve a Java FQN and instantiate it via reflection.

    Requires the class to be compiled with -parameters so parameter names are
    retained in the class file. With empty strategy_args, uses the no-arg constructor.
    """
    try:
        cls = jpype.JClass(class_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Java strategy class {class_name!r}. "
            "Make sure the JAR containing it is on the JVM classpath "
            "(set GNOME_JARS, pass extra_jars=, or use --jar)."
        ) from e

    if not strategy_args:
        try:
            return cls()
        except Exception as e:
            raise RuntimeError(
                f"{class_name} has no no-arg constructor; "
                "supply strategy_args matching one of its constructors."
            ) from e

    requested = set(strategy_args.keys())
    constructors = list(cls.class_.getConstructors())
    candidates = []
    for ctor in constructors:
        params = list(ctor.getParameters())
        names = [str(p.getName()) for p in params]
        if set(names) == requested:
            candidates.append((ctor, names))

    if not candidates:
        available = [
            [str(p.getName()) for p in ctor.getParameters()] for ctor in constructors
        ]
        raise RuntimeError(
            f"No constructor on {class_name} matches keys "
            f"{sorted(requested)}. Available constructors: {available}. "
            "If parameter names show as 'arg0/arg1/...', the class was "
            "compiled without -parameters; enable it in the maven-compiler-plugin."
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Ambiguous: multiple constructors on {class_name} match "
            f"{sorted(requested)}"
        )

    ctor, names = candidates[0]
    ordered = [strategy_args[n] for n in names]
    return ctor.newInstance(ordered)


def _load_python_strategy(import_path: str, kwargs: dict | None = None) -> Strategy:
    """Resolve a 'module.path:ClassName' import path and instantiate it."""
    if ":" not in import_path:
        raise ValueError(f"strategy must be 'module.path:ClassName', got: {import_path!r}")
    module_path, class_name = import_path.split(":", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    instance = cls(**(kwargs or {}))
    if not isinstance(instance, Strategy):
        raise ValueError(
            f"{import_path} did not produce a gnomepy.Strategy instance "
            f"(got {type(instance).__name__})"
        )
    return instance


class Backtest:
    """Orchestrate a backtest with a Python or Java strategy against Java simulation.

    Usage::

        # With a YAML config file
        results = Backtest("config.yaml", strategy=MyStrategy()).run()

        # With a programmatic config
        config = BacktestConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 2),
            listings=[ListingSimConfig(listing_id=1, profile="default")],
            profiles={"default": ExchangeProfileConfig()},
        )
        results = Backtest(config, strategy=MyStrategy()).run()
    """

    def __init__(
        self,
        config: BacktestConfig | str | Path,
        strategy: Strategy | str | None = None,
        *,
        backtest_id: str | None = None,
        registry_url: str | None = None,
        registry_api_key: str | None = None,
        s3_client=None,
        strategy_args: dict | None = None,
        original_config_path: str | Path | None = None,
        cache: bool | str | Path = True,
    ):
        """
        Args:
            config: Python BacktestConfig, or path to a YAML config file.
            strategy: Python Strategy instance, Java FQN string, Python import path
                "module:ClassName", or None to use strategy from YAML config.
            registry_url: Registry API URL. Defaults to env-based config.
            registry_api_key: Registry API key. Defaults to GNOME_REGISTRY_API_KEY env var.
            s3_client: Pre-built Java S3Client. Created automatically if omitted.
            strategy_args: Constructor args for Java strategies or Python strategies
                resolved from a YAML import path.
            original_config_path: Override for config_path in metadata. Useful when config
                is a temp file (e.g. during parameter sweeps) and the original path should
                be recorded instead.
            cache: Enable local market data caching. True (default) uses ~/.gnomepy/cache/,
                a path string uses that directory, False disables caching.
        """
        ensure_jvm_started()
        self._config = config
        self._strategy = strategy
        self._backtest_id = backtest_id
        self._registry_url = registry_url
        self._registry_api_key = registry_api_key
        self._s3_client = s3_client
        self._strategy_args = strategy_args or {}
        self._cache = cache
        self._original_config_path = original_config_path
        self._recorder = None
        self._driver = None
        self._start_date = None
        self._end_date = None
        self._warnings: list[str] = []
        self._warning_handler = None

    def _build_driver(self):
        BacktestDriverFactory = jpype.JClass(
            "group.gnometrading.backtest.config.BacktestDriverFactory"
        )
        JavaBacktestConfig = jpype.JClass("group.gnometrading.backtest.config.BacktestConfig")
        Paths = jpype.JClass("java.nio.file.Paths")

        if isinstance(self._config, (str, Path)):
            java_config = JavaBacktestConfig.fromYaml(Paths.get(str(self._config)))
        else:
            java_config = self._config._to_java()

        # Extract dates for progress reporting
        start = java_config.startDate
        end = java_config.endDate
        self._start_date = datetime(
            int(start.getYear()), int(start.getMonthValue()), int(start.getDayOfMonth()),
            int(start.getHour()), int(start.getMinute()), int(start.getSecond()),
        )
        self._end_date = datetime(
            int(end.getYear()), int(end.getMonthValue()), int(end.getDayOfMonth()),
            int(end.getHour()), int(end.getMinute()), int(end.getSecond()),
        )

        # Build SecurityMaster
        registry_host = self._registry_url or os.environ.get("REGISTRY_URL", gnome_config.REGISTRY_API_HOST)
        registry_api_key = self._registry_api_key or os.environ.get("REGISTRY_API_KEY", gnome_config.REGISTRY_API_KEY)
        RegistryConnection = jpype.JClass("group.gnometrading.RegistryConnection")
        SecurityMaster = jpype.JClass("group.gnometrading.SecurityMaster")
        security_master = SecurityMaster(RegistryConnection(registry_host, registry_api_key))

        java_oms = BacktestDriverFactory.buildOms(java_config.risk, security_master)

        if java_config.record:
            self._recorder = jpype.JClass(
                "group.gnometrading.backtest.recorder.BacktestRecorder"
            )(jpype.JInt(int(java_config.recordDepth)))

        java_strategy = self._resolve_strategy(java_config, java_oms, security_master)

        if self._s3_client is None:
            s3 = jpype.JClass("software.amazon.awssdk.services.s3.S3Client").create()
        else:
            s3 = self._s3_client

        if self._cache is not False:
            from gnomepy.java.cache import MarketDataCache, create_caching_s3_proxy
            cache_dir = self._cache if isinstance(self._cache, (str, Path)) else None
            md_cache = MarketDataCache(cache_dir)
            bucket = f"gnome-market-data-{os.getenv('STAGE', 'prod').lower()}"
            s3 = create_caching_s3_proxy(s3, md_cache, bucket)
            logger.debug("market data caching enabled: %s", md_cache._root)

        self._driver = BacktestDriverFactory.create(
            java_config, security_master, java_oms, java_strategy, self._recorder, s3
        )

    def _resolve_strategy(self, java_config, java_oms, security_master):
        position_view = java_oms.getPositionTracker().createPositionView(jpype.JInt(0))
        PythonStrategyAgent = jpype.JClass(
            "group.gnometrading.backtest.driver.PythonStrategyAgent"
        )
        strategy = self._strategy

        if strategy is None:
            if java_config.strategy is None:
                raise ValueError(
                    "No strategy provided and config has no strategy.class_name"
                )
            class_name = str(java_config.strategy.className)
            args = {}
            if java_config.strategy.args is not None:
                args = {str(k): v for k, v in dict(java_config.strategy.args).items()}
            if ":" in class_name:
                py_strategy = _load_python_strategy(class_name, args)
                return self._wrap_python_strategy(py_strategy, java_oms, security_master, position_view, PythonStrategyAgent)
            return _instantiate_java_strategy(class_name, args)

        if isinstance(strategy, str) and ":" in strategy:
            py_strategy = _load_python_strategy(strategy, self._strategy_args)
            return self._wrap_python_strategy(py_strategy, java_oms, security_master, position_view, PythonStrategyAgent)

        if isinstance(strategy, str):
            return _instantiate_java_strategy(strategy, self._strategy_args)

        return self._wrap_python_strategy(strategy, java_oms, security_master, position_view, PythonStrategyAgent)

    def _wrap_python_strategy(self, py_strategy, java_oms, security_master, position_view, PythonStrategyAgent):
        py_strategy._oms_view = OmsView(java_oms, security_master)
        if self._recorder is not None:
            py_strategy._metric_recorder = PyMetricRecorder(self._recorder.createMetricRecorder())
            py_strategy.register_metrics()
        callback = _create_python_callback(py_strategy)
        return PythonStrategyAgent.create(position_view, callback)

    def add_warning(self, message: str) -> None:
        """Add an arbitrary warning to be included in backtest results and metadata."""
        self._warnings.append(message)

    def _install_warning_handler(self) -> None:
        WarningHandler = jpype.JClass("group.gnometrading.backtest.recorder.WarningHandler")
        self._warning_handler = WarningHandler()
        jpype.JClass("java.util.logging.Logger").getLogger("group.gnometrading").addHandler(self._warning_handler)

    def _collect_java_warnings(self) -> None:
        if self._warning_handler is None:
            return
        for msg in self._warning_handler.getMessages():
            self.add_warning(str(msg))
        self._warning_handler.clearMessages()

    def run(self, progress: bool = True) -> BacktestResults | None:
        """Prepare data and fully execute the backtest."""
        t0 = time.time()
        self._build_driver()
        self._install_warning_handler()
        self._driver.prepareData()

        if not progress:
            self._driver.fullyExecute()
        else:
            self._run_with_progress()

        self._collect_java_warnings()
        wall_time = time.time() - t0
        event_count = int(self._driver.getEventsProcessed())

        if self._recorder is not None:
            metadata = self._build_metadata(wall_time=wall_time, event_count=event_count)
            return BacktestResults(self._recorder, metadata=metadata)
        return None

    def _resolve_strategy_name(self) -> str | None:
        """Extract a human-readable strategy name from self._strategy."""
        if isinstance(self._strategy, str):
            return self._strategy
        if self._strategy is not None:
            cls = type(self._strategy)
            return f"{cls.__module__}:{cls.__name__}"
        return None

    def _build_metadata(self, wall_time: float, event_count: int) -> BacktestMetadata:
        """Assemble BacktestMetadata from all available sources."""
        strategy_name = self._resolve_strategy_name()

        if self._backtest_id is None:
            self._backtest_id = generate_backtest_id(strategy_name)

        if self._original_config_path is not None:
            config_path = str(self._original_config_path)
        elif isinstance(self._config, (str, Path)):
            config_path = str(self._config)
        else:
            config_path = None

        try:
            gnomepy_version = _pkg_version("gnomepy")
        except Exception:
            gnomepy_version = None

        try:
            gnomepy_research_version = _pkg_version("gnomepy_research")
        except Exception:
            gnomepy_research_version = None

        gnomepy_research_commit = os.environ.get("RESEARCH_COMMIT")

        return BacktestMetadata(
            backtest_id=self._backtest_id,
            start_date=str(self._start_date) if self._start_date else None,
            end_date=str(self._end_date) if self._end_date else None,
            wall_time_seconds=round(wall_time, 3),
            event_count=event_count,
            strategy=strategy_name,
            strategy_args=self._strategy_args or None,
            config_path=config_path,
            preset_name=getattr(self, "_preset_name", None),
            config=getattr(self, "_preset_config", None),
            gnomepy_version=gnomepy_version,
            gnomepy_research_version=gnomepy_research_version,
            gnomepy_research_commit=gnomepy_research_commit,
            warnings=self._warnings,
        )

    def _run_with_progress(self):
        start = self._start_date
        end = self._end_date
        total_sec = (end - start).total_seconds()
        if total_sec <= 0:
            self._driver.fullyExecute()
            return

        epoch = datetime(1970, 1, 1, tzinfo=pytz.UTC)
        start_ns = int((start.replace(tzinfo=pytz.UTC) - epoch).total_seconds() * 1_000_000_000)
        end_ns = int((end.replace(tzinfo=pytz.UTC) - epoch).total_seconds() * 1_000_000_000)

        chunk_ns = 10 * 60_000_000_000 # 10 minutes
        current_ns = start_ns
        t0 = time.time()

        while current_ns < end_ns:
            current_ns += chunk_ns
            self._driver.executeUntil(jpype.JLong(min(current_ns, end_ns)))
            elapsed = time.time() - t0
            pct = min((current_ns - start_ns) / (end_ns - start_ns) * 100, 100)
            events = int(self._driver.getEventsProcessed())
            logger.info("Backtest: %d%% | events: %s | %.1fs", pct, f"{events:,}", elapsed)

        self._driver.fullyExecute()
        elapsed = time.time() - t0
        events = int(self._driver.getEventsProcessed())
        logger.info("Backtest: 100%% | events: %s | %.1fs", f"{events:,}", elapsed)

    def run_until(self, timestamp: int) -> BacktestResults | None:
        """Run the backtest until a specific nanosecond timestamp."""
        if self._driver is None:
            self._build_driver()
            self._install_warning_handler()
            self._driver.prepareData()
        self._driver.executeUntil(jpype.JLong(timestamp))
        self._collect_java_warnings()
        if self._recorder is not None:
            metadata = self._build_metadata(
                wall_time=0,
                event_count=int(self._driver.getEventsProcessed()),
            )
            return BacktestResults(self._recorder, metadata=metadata)
        return None


def run_backtest(
    config: BacktestConfig | str | Path,
    strategy: Strategy | str | None = None,
    *,
    backtest_id: str | None = None,
    registry_url: str | None = None,
    registry_api_key: str | None = None,
    s3_client=None,
    strategy_args: dict | None = None,
    original_config_path: str | Path | None = None,
    progress: bool = True,
    cache: bool | str | Path = True,
) -> BacktestResults | None:
    """Run a backtest end-to-end.

    Args:
        config: Python BacktestConfig or path to a YAML config file.
        strategy: Python Strategy instance, Java FQN, Python "module:Class" import path,
            or None to use strategy from YAML config.
        backtest_id: Optional explicit ID for this run. Auto-generated if omitted.
        original_config_path: Override for config_path in metadata (useful when config
            is a temp file during parameter sweeps).
        cache: Enable local market data caching. True uses ~/.gnomepy/cache/, a path
            uses that directory, False disables caching.

    Returns BacktestResults if recording is enabled, else None.
    """
    bt = Backtest(
        config,
        strategy,
        backtest_id=backtest_id,
        registry_url=registry_url,
        registry_api_key=registry_api_key,
        s3_client=s3_client,
        strategy_args=strategy_args,
        original_config_path=original_config_path,
        cache=cache,
    )
    return bt.run(progress=progress)
