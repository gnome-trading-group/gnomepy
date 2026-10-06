"""Backtest runner plumbing that needs a JVM — requires the gnome-backtest uber JAR; skipped without it."""
from __future__ import annotations

from datetime import date

import jpype
import pytest

from gnomepy.java._classpath import discover_classpath
from gnomepy.java._jvm import ensure_jvm_started
from gnomepy.java.backtest.config import (
    BacktestConfig,
    ExchangeProfileConfig,
    ListingSimConfig,
    RiskConfig,
    PolicyConfig,
)
from gnomepy.java.backtest.runner import Backtest, _to_java_value, _to_python
from gnomepy.java.backtest.strategy import Strategy


def _jar_available() -> bool:
    try:
        return len(discover_classpath("gnome-backtest")) > 0
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not _jar_available(), reason="gnome-backtest JAR not available")


@pytest.fixture(scope="module", autouse=True)
def _jvm():
    ensure_jvm_started()


def _backtest() -> Backtest:
    config = BacktestConfig(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 2),
        listings=[ListingSimConfig(listing_id=1, profile="default")],
        profiles={"default": ExchangeProfileConfig()},
        record=False,
    )
    return Backtest(config)


def test_yaml_args_become_python_types():
    HashMap = jpype.JClass("java.util.HashMap")
    ArrayList = jpype.JClass("java.util.ArrayList")
    inner = HashMap()
    inner.put("depth", jpype.JClass("java.lang.Integer").valueOf(3))
    values = ArrayList()
    values.add(jpype.JClass("java.lang.Double").valueOf(0.5))
    values.add(jpype.JString("x"))

    assert _to_python(jpype.JString("abc")) == "abc"
    assert type(_to_python(jpype.JString("abc"))) is str
    assert _to_python(jpype.JClass("java.lang.Long").valueOf(7)) == 7
    assert type(_to_python(jpype.JClass("java.lang.Integer").valueOf(7))) is int
    assert _to_python(jpype.JClass("java.lang.Boolean").valueOf(True)) is True
    assert _to_python(inner) == {"depth": 3}
    assert _to_python(values) == [0.5, "x"]


def test_python_args_become_java_values():
    converted = _to_java_value({"depth": 3, "edge": 0.5, "on": True, "levels": [1, 2], "name": "mm"})

    assert str(converted.get("depth").getClass().getName()) == "java.lang.Long"
    assert str(converted.get("edge").getClass().getName()) == "java.lang.Double"
    assert str(converted.get("on").getClass().getName()) == "java.lang.Boolean"
    assert str(converted.get("levels").getClass().getName()) == "java.util.ArrayList"
    assert converted.get("name") == "mm"


def test_strategy_id_and_policies_reach_the_java_config():
    config = BacktestConfig(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 2),
        listings=[ListingSimConfig(listing_id=1, profile="default")],
        profiles={"default": ExchangeProfileConfig()},
        strategy_id=12,
        risk=RiskConfig(
            from_registry=True,
            policies=[PolicyConfig(type="MAX_ORDER_SIZE", listing_id=1, params={"maxOrderSize": 5})],
        ),
    )
    java = config._to_java()

    assert int(java.strategyId) == 12
    assert bool(java.risk.fromRegistry)
    scoped = java.risk.policies.get(0)
    assert str(scoped.type) == "MAX_ORDER_SIZE"
    assert int(scoped.listingId) == 1
    assert int(scoped.strategyId) == 0


def test_warning_handler_is_removed_from_the_global_logger():
    logger = jpype.JClass("java.util.logging.Logger").getLogger("group.gnometrading")
    before = len(logger.getHandlers())
    backtest = _backtest()

    backtest._install_warning_handler()
    backtest._install_warning_handler()
    assert len(logger.getHandlers()) == before + 1

    backtest._remove_warning_handler()
    assert len(logger.getHandlers()) == before


class _MetricsStrategy(Strategy):
    def __init__(self):
        self.saw_positions = None

    def register_metrics(self):
        self.saw_positions = self.positions is not None
        buf = self.metrics.create_buffer("signals")
        self.col = buf.addDoubleColumn("value")
        buf.freeze()
        self.buf = buf

    def on_market_data(self, data):
        return []

    def on_execution_report(self, report):
        return []


def test_register_metrics_runs_without_recording_and_after_init():
    strategy = _MetricsStrategy()

    _backtest()._wrap_python_strategy(strategy, None, None, 0)

    assert strategy.saw_positions is True
    strategy.buf.setDouble(strategy.buf.appendRow(), strategy.col, 1.0)
    assert int(strategy.buf.getCount()) == 0
