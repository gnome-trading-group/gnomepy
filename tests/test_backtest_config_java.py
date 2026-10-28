"""Checks that backtest config fields reach their Java counterparts.

Requires the gnome-backtest uber JAR (GNOME_JARS or the sibling-directory
discovery path). Skipped when it can't be found.
"""
from __future__ import annotations

from datetime import date

import jpype
import pytest

from gnomepy.java._classpath import discover_classpath
from gnomepy.java._jvm import ensure_jvm_started
from gnomepy.java.backtest.config import (
    BacktestConfig,
    OptimisticQueueConfig,
    SimulationOverrides,
    LogNormalLatencyConfig,
    RecordedLatencyConfig,
    ListingSimConfig,
)


def _jar_available() -> bool:
    try:
        return len(discover_classpath("gnome-backtest")) > 0
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not _jar_available(), reason="gnome-backtest JAR not available")


@pytest.fixture(scope="module", autouse=True)
def _jvm():
    ensure_jvm_started()


def _config(**kwargs) -> BacktestConfig:
    kwargs.setdefault("listings", [ListingSimConfig(listing_id=1)])
    return BacktestConfig(start_date=date(2026, 1, 1), end_date=date(2026, 1, 2), **kwargs)


def test_defaults_leave_java_defaults_in_place():
    java = _config()._to_java()
    JavaBacktestConfig = jpype.JClass("group.gnometrading.backtest.config.BacktestConfig")
    assert java.measureProcessingTime is False
    assert int(java.seed) == int(JavaBacktestConfig.DEFAULT_SEED)


def test_measure_processing_time_and_seed_reach_java():
    java = _config(measure_processing_time=True, seed=1234)._to_java()
    assert java.measureProcessingTime is True
    assert int(java.seed) == 1234


def test_on_missing_reaches_java():
    assert str(_config()._to_java().onMissing.name()) == "MARK"
    assert str(_config(on_missing="fail")._to_java().onMissing.name()) == "FAIL"


def test_lognormal_seed_is_optional():
    assert LogNormalLatencyConfig()._to_java().seed is None
    assert int(LogNormalLatencyConfig(seed=7)._to_java().seed) == 7


def test_latency_configs_reach_java():
    java = SimulationOverrides(
        market_data_latency=RecordedLatencyConfig(fallback=LogNormalLatencyConfig(floor_nanos=1)),
        network_latency=LogNormalLatencyConfig(floor_nanos=2, median_nanos=3, p99_nanos=4),
    )._to_java()
    assert int(java.marketDataLatency.fallback.floorNanos) == 1
    assert (int(java.networkLatency.floorNanos), int(java.networkLatency.medianNanos)) == (2, 3)
    assert int(java.networkLatency.p99Nanos) == 4


def test_self_trade_prevention_reaches_java():
    overrides = SimulationOverrides(self_trade_prevention="CANCEL_RESTING")._to_java()
    assert str(overrides.selfTradePrevention.name()) == "CANCEL_RESTING"
    assert SimulationOverrides()._to_java().selfTradePrevention is None


def test_unset_blocks_stay_unset_in_java_so_the_registry_fills_them():
    java = SimulationOverrides(queue_model=OptimisticQueueConfig())._to_java()
    assert java.networkLatency is None
    assert java.feeModel is None
    assert java.queueModel is not None


def test_exchange_and_listing_overrides_and_the_pinned_time_reach_java():
    java = _config(
        exchanges={"KALSHI": SimulationOverrides(queue_model=OptimisticQueueConfig())},
        simulation_as_of="2026-10-01T00:00:00Z",
        listings=[ListingSimConfig(listing_id=1, overrides=SimulationOverrides(self_trade_prevention="CANCEL_RESTING"))],
    )._to_java()
    assert java.exchanges.get("KALSHI").queueModel is not None
    assert str(java.simulationAsOf) == "2026-10-01T00:00:00Z"
    assert str(java.listings.get(0).overrides.selfTradePrevention.name()) == "CANCEL_RESTING"
