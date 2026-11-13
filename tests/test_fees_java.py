"""FeeSchedule conversion from Java — requires the gnome-backtest uber JAR; skipped without it."""
from __future__ import annotations

import jpype
import pytest

from gnomepy.fees import FeeSchedule
from gnomepy.java._classpath import discover_classpath
from gnomepy.java._jvm import ensure_jvm_started

FEE = "group.gnometrading.sm.fee."


def _jar_available() -> bool:
    try:
        return len(discover_classpath("gnome-backtest")) > 0
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not _jar_available(), reason="gnome-backtest JAR not available")


@pytest.fixture(scope="module", autouse=True)
def _jvm():
    ensure_jvm_started()


def _java_schedule(rebate: float):
    component = jpype.JClass(FEE + "FeeComponent")(
        jpype.JClass(FEE + "FeeKind").PRICE_CURVE, 0.07, 0.0175, 1.0, jpype.JClass(FEE + "FeeSide").BOTH)
    components = jpype.JArray(jpype.JClass(FEE + "FeeComponent"))([component])
    rounding = jpype.JClass(FEE + "FeeRounding").NONE
    return jpype.JClass(FEE + "FeeSchedule")("USD", components, rounding, rebate)


def test_from_java_carries_the_maker_rebate_share():
    schedule = FeeSchedule.from_java(_java_schedule(0.25))
    assert schedule.maker_rebate_share == 0.25
    assert schedule.price_curve_rates() == (0.07, 0.0175)


def test_from_java_without_a_rebate_reads_zero():
    assert FeeSchedule.from_java(_java_schedule(0.0)).maker_rebate_share == 0.0
