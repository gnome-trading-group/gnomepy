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
    ExchangeProfileConfig,
    GaussianLatencyConfig,
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
    return BacktestConfig(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 2),
        listings=[ListingSimConfig(listing_id=1, profile="default")],
        profiles={"default": ExchangeProfileConfig()},
        **kwargs,
    )


def test_defaults_leave_java_defaults_in_place():
    java = _config()._to_java()
    JavaBacktestConfig = jpype.JClass("group.gnometrading.backtest.config.BacktestConfig")
    assert java.measureProcessingTime is False
    assert int(java.seed) == int(JavaBacktestConfig.DEFAULT_SEED)


def test_measure_processing_time_and_seed_reach_java():
    java = _config(measure_processing_time=True, seed=1234)._to_java()
    assert java.measureProcessingTime is True
    assert int(java.seed) == 1234


def test_gaussian_seed_is_optional():
    assert GaussianLatencyConfig(mu=1e6, sigma=1e5)._to_java().seed is None
    assert int(GaussianLatencyConfig(mu=1e6, sigma=1e5, seed=7)._to_java().seed) == 7
