from __future__ import annotations

import hashlib
import importlib
import logging
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml

import jpype
import pytz
from jpype import JImplements, JOverride

from importlib.metadata import version as _pkg_version

logger = logging.getLogger(__name__)

from gnomepy.config import config as gnome_config, resolve_registry_api_key
from gnomepy.java._jvm import ensure_jvm_started
from gnomepy.java.backtest.config import BacktestConfig
from gnomepy.java.backtest.orders import ExecutionReport
from gnomepy.java.backtest.strategy import Strategy
from gnomepy.java.oms import PositionViewWrapper
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
        "group.gnometrading.strategies.PythonStrategyAgent$PythonStrategyCallback"
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

        @JOverride
        def onInit(self, positionView, securityMaster):
            py_strategy._position_view = PositionViewWrapper(positionView, securityMaster)

    return _Proxy()


def _to_python(value):
    """Converts a value Jackson read from YAML (Java String, boxed number, List, Map) to its Python type."""
    # Java checks first: JPype's boxed numbers and strings subclass Python's own types.
    if isinstance(value, jpype.JString):
        return str(value)
    if isinstance(value, jpype.JClass("java.lang.Boolean")):
        return bool(value)
    if isinstance(value, (jpype.JClass("java.lang.Double"), jpype.JClass("java.lang.Float"))):
        return float(value)
    if isinstance(value, jpype.JClass("java.lang.Number")):
        return int(value)
    if isinstance(value, jpype.JClass("java.util.Map")):
        return {str(k): _to_python(v) for k, v in dict(value).items()}
    if isinstance(value, jpype.JClass("java.util.List")):
        return [_to_python(v) for v in value]
    return value


def _to_java_value(value):
    """Converts a Python strategy argument to the Java value StrategyFactory matches to a constructor parameter."""
    # bool first: it subclasses int.
    if isinstance(value, bool):
        return jpype.JClass("java.lang.Boolean").valueOf(value)
    if isinstance(value, int):
        return jpype.JClass("java.lang.Long").valueOf(value)
    if isinstance(value, float):
        return jpype.JClass("java.lang.Double").valueOf(value)
    if isinstance(value, dict):
        result = jpype.JClass("java.util.LinkedHashMap")()
        for k, v in value.items():
            result.put(str(k), _to_java_value(v))
        return result
    if isinstance(value, (list, tuple)):
        result = jpype.JClass("java.util.ArrayList")()
        for v in value:
            result.add(_to_java_value(v))
        return result
    return value


def _create_java_strategy(class_name: str, strategy_id: int, position_view, security_master, args):
    """Builds a Java strategy as the orchestrator does live: infrastructure first, then ``args`` by parameter name."""
    StrategyFactory = jpype.JClass("group.gnometrading.strategies.StrategyFactory")
    try:
        jpype.JClass(class_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Java strategy class {class_name!r}. "
            "Make sure the JAR containing it is on the JVM classpath "
            "(set GNOME_JARS, pass extra_jars=, or use --jar)."
        ) from e
    return StrategyFactory.createWithOwnBuffers(
        class_name, jpype.JInt(strategy_id), position_view, security_master, args
    )


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
        registry_api_key = self._registry_api_key or os.environ.get("REGISTRY_API_KEY") or resolve_registry_api_key()
        RegistryConnection = jpype.JClass("group.gnometrading.RegistryConnection")
        SecurityMaster = jpype.JClass("group.gnometrading.SecurityMaster")
        registry = RegistryConnection(registry_host, registry_api_key)
        security_master = SecurityMaster(registry)

        context = BacktestDriverFactory.buildContext(java_config)
        java_oms = BacktestDriverFactory.buildOms(java_config, security_master, registry, context)

        strategy_id = int(java_config.strategyId)
        tracker = java_oms.getPositionTracker()
        for lsc in java_config.listings:
            tracker.registerSlot(jpype.JInt(strategy_id), jpype.JInt(int(lsc.listingId)))

        if java_config.record:
            self._recorder = jpype.JClass(
                "group.gnometrading.backtest.recorder.BacktestRecorder"
            )(jpype.JInt(int(java_config.recordDepth)))

        java_strategy = self._resolve_strategy(java_config, java_oms, security_master, strategy_id)

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
            java_config, security_master, java_oms, java_strategy, self._recorder, s3, context
        )

    def _resolve_strategy(self, java_config, java_oms, security_master, strategy_id):
        position_view = java_oms.getPositionTracker().createPositionView(jpype.JInt(strategy_id))
        strategy = self._strategy

        if strategy is None:
            if java_config.strategy is None:
                raise ValueError(
                    "No strategy provided and config has no strategy.class_name"
                )
            class_name = str(java_config.strategy.className)
            java_args = java_config.strategy.args
            if ":" in class_name:
                args = {str(k): _to_python(v) for k, v in dict(java_args).items()} if java_args is not None else {}
                py_strategy = _load_python_strategy(class_name, args)
                return self._wrap_python_strategy(py_strategy, security_master, position_view, strategy_id)
            if java_args is None:
                java_args = jpype.JClass("java.util.HashMap")()
            return _create_java_strategy(class_name, strategy_id, position_view, security_master, java_args)

        if isinstance(strategy, str) and ":" in strategy:
            py_strategy = _load_python_strategy(strategy, self._strategy_args)
            return self._wrap_python_strategy(py_strategy, security_master, position_view, strategy_id)

        if isinstance(strategy, str):
            return _create_java_strategy(
                strategy, strategy_id, position_view, security_master, _to_java_value(self._strategy_args)
            )

        return self._wrap_python_strategy(strategy, security_master, position_view, strategy_id)

    def _wrap_python_strategy(self, py_strategy, security_master, position_view, strategy_id):
        PythonStrategyAgent = jpype.JClass("group.gnometrading.strategies.PythonStrategyAgent")
        if self._recorder is not None:
            java_metrics = self._recorder.createMetricRecorder()
        else:
            # Strategies declare and write metrics the same way whether or not the run records them.
            java_metrics = jpype.JClass("group.gnometrading.backtest.recorder.MetricRecorder").discarding()
        py_strategy._metric_recorder = PyMetricRecorder(java_metrics)
        callback = _create_python_callback(py_strategy)
        agent = PythonStrategyAgent.create(jpype.JInt(strategy_id), position_view, security_master, callback)
        # After create, which runs onInit, so self.positions is available inside register_metrics.
        py_strategy.register_metrics()
        return agent

    def add_warning(self, message: str) -> None:
        """Add an arbitrary warning to be included in backtest results and metadata."""
        self._warnings.append(message)

    def _install_warning_handler(self) -> None:
        if self._warning_handler is not None:
            return
        WarningHandler = jpype.JClass("group.gnometrading.backtest.recorder.WarningHandler")
        self._warning_handler = WarningHandler()
        jpype.JClass("java.util.logging.Logger").getLogger("group.gnometrading").addHandler(self._warning_handler)

    def _remove_warning_handler(self) -> None:
        # The logger is global to the JVM: a handler left on it would collect every later run's warnings too.
        if self._warning_handler is None:
            return
        jpype.JClass("java.util.logging.Logger").getLogger("group.gnometrading").removeHandler(self._warning_handler)
        self._warning_handler = None

    def _collect_java_warnings(self) -> None:
        if self._warning_handler is None:
            return
        for msg in self._warning_handler.getMessages():
            self.add_warning(str(msg))
        self._warning_handler.clearMessages()

    def run(self, progress: bool = True) -> BacktestResults | None:
        """Prepare data and fully execute the backtest."""
        t0 = time.time()
        # Before the build, so warnings raised while building are kept too.
        self._install_warning_handler()
        try:
            self._build_driver()
            self._driver.prepareData()

            end_ns = int(self._end_date.replace(tzinfo=pytz.UTC).timestamp()) * 1_000_000_000
            if not progress:
                self._driver.executeUntil(jpype.JLong(end_ns))
            else:
                self._run_with_progress()

            self._collect_java_warnings()
        finally:
            self._remove_warning_handler()
        if self._recorder is not None:
            self._recorder.closeOpenOrders(jpype.JLong(end_ns))
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

    def _capture_env_metadata(self) -> dict:
        """Capture reproducibility metadata: git commit, JAR hash, runtime versions."""
        result = {}

        try:
            repo_root = Path(__file__).parents[3]
            proc = subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True,
            )
            if proc.returncode == 0:
                result["gnomepy_commit"] = proc.stdout.strip()
        except Exception:
            pass

        try:
            import jpype as _jpype
            result["java_version"] = str(_jpype.getJVMVersion())
        except Exception:
            pass

        result["python_version"] = platform.python_version()
        result["os_info"] = platform.platform()

        gnome_jars = os.environ.get("GNOME_JARS", "")
        if gnome_jars:
            try:
                jar_path = Path(gnome_jars.split(":")[0])
                digest = hashlib.sha256(jar_path.read_bytes()).hexdigest()[:16]
                result["backtest_jar_hash"] = f"sha256:{digest}"
            except Exception:
                pass

        return result

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

        env = self._capture_env_metadata()

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
            config=getattr(self, "_preset_config", None) or (
                yaml.safe_load(Path(self._config).read_text())
                if isinstance(self._config, (str, Path)) else None
            ),
            gnomepy_version=gnomepy_version,
            gnomepy_research_version=gnomepy_research_version,
            gnomepy_research_commit=gnomepy_research_commit,
            gnomepy_commit=env.get("gnomepy_commit"),
            backtest_jar_hash=env.get("backtest_jar_hash"),
            java_version=env.get("java_version"),
            python_version=env.get("python_version"),
            os_info=env.get("os_info"),
            warnings=self._warnings,
            oms_rejects=(
                {str(k): int(v) for k, v in dict(self._recorder.getOmsRejectCounts()).items()}
                if self._recorder is not None
                else {}
            ),
        )

    def _run_with_progress(self):
        start = self._start_date
        end = self._end_date
        total_sec = (end - start).total_seconds()
        if total_sec <= 0:
            return

        start_ns = int(start.replace(tzinfo=pytz.UTC).timestamp()) * 1_000_000_000
        end_ns = int(end.replace(tzinfo=pytz.UTC).timestamp()) * 1_000_000_000

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

        elapsed = time.time() - t0
        events = int(self._driver.getEventsProcessed())
        logger.info("Backtest: 100%% | events: %s | %.1fs", f"{events:,}", elapsed)

    def run_until(self, timestamp: int) -> BacktestResults | None:
        """Run the backtest until a specific nanosecond timestamp."""
        if self._driver is None:
            self._install_warning_handler()
            self._build_driver()
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
