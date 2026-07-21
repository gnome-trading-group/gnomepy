from __future__ import annotations

import importlib
import signal
import time
from pathlib import Path
from typing import Union

import traceback

import jpype
from jpype import JImplements, JOverride

from gnomepy.java._classpath import discover_classpath
from gnomepy.java._jvm import ensure_jvm_started
from gnomepy.java.backtest.strategy import Strategy
from gnomepy.java.oms import PositionViewWrapper
from gnomepy.java.schemas import wrap_schema
from gnomepy.java.strategy.config import SessionConfig, StrategyConfig


def _load_python_strategy(import_path: str, kwargs: dict | None = None) -> Strategy:
    if ":" not in import_path:
        raise ValueError(f"Python strategy must be 'module.path:ClassName', got: {import_path!r}")
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


def _create_strategy_callback(py_strategy: Strategy):
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
            try:
                wrapped = wrap_schema(data)
                intents = py_strategy.on_market_data(wrapped)
                return _to_java_list(intents)
            except Exception:
                traceback.print_exc()
                raise

        @JOverride
        def onExecutionReport(self, report):
            try:
                from gnomepy.java.backtest.orders import ExecutionReport
                py_report = ExecutionReport._from_java(report)
                intents = py_strategy.on_execution_report(py_report)
                return _to_java_list(intents)
            except Exception:
                traceback.print_exc()
                raise

        @JOverride
        def simulateProcessingTime(self):
            return jpype.JLong(py_strategy.simulate_processing_time())

        @JOverride
        def onInit(self, positionView, securityMaster):
            py_strategy._position_view = PositionViewWrapper(positionView, securityMaster)

    return _Proxy()


def run_strategy_session(
    config: SessionConfig,
    strategy: Union[str, Strategy, None] = None,
    jar: str | None = None,
    gnome_root: str | Path | None = None,
) -> None:
    """Run a strategy session locally against live market feeds.

    Blocks until the JVM shuts down (SIGINT/SIGTERM or explicit shutdown).

    Args:
        config: Session configuration parsed from YAML.
        strategy: Python Strategy instance or 'module:ClassName' import path.
            If None, uses config.strategy.class_name.
        jar: Explicit path to the gnome-orchestrator uber JAR.
        gnome_root: Root directory of GNOME repos for JAR discovery.
    """
    classpath = [jar] if jar else discover_classpath("gnome-orchestrator", gnome_root)
    ensure_jvm_started(classpath=classpath)

    effective_strategy = strategy
    if effective_strategy is None and config.strategy is not None:
        effective_strategy = config.strategy.class_name

    if isinstance(effective_strategy, str) and ":" in effective_strategy:
        strategy_args = config.strategy.args if config.strategy else {}
        py_strategy = _load_python_strategy(effective_strategy, strategy_args)
        callback = _create_strategy_callback(py_strategy)
        PythonStrategyAgent = jpype.JClass("group.gnometrading.strategies.PythonStrategyAgent")
        PythonStrategyAgent.setCallback(callback)
    elif isinstance(effective_strategy, Strategy):
        callback = _create_strategy_callback(effective_strategy)
        PythonStrategyAgent = jpype.JClass("group.gnometrading.strategies.PythonStrategyAgent")
        PythonStrategyAgent.setCallback(callback)

    props = config.to_properties()
    java_args = jpype.JArray(jpype.JString)([f"--{k}={v}" for k, v in props.items()])

    System = jpype.JClass("java.lang.System")
    jpype.JClass("group.gnometrading.trading.TradingOrchestrator")
    Orchestrator = jpype.JClass("group.gnometrading.di.Orchestrator")
    Orchestrator.main(java_args)

    def _shutdown(sig, frame):
        System.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        time.sleep(1)
