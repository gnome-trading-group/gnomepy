"""ExecutionReport conversion from Java — requires the gnome-backtest uber JAR; skipped without it."""
from __future__ import annotations

import jpype
import pytest

from gnomepy.java._classpath import discover_classpath
from gnomepy.java._jvm import ensure_jvm_started
from gnomepy.java.backtest.orders import ExecutionReport
from gnomepy.java.enums import ExecType


def _jar_available() -> bool:
    try:
        return len(discover_classpath("gnome-backtest")) > 0
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not _jar_available(), reason="gnome-backtest JAR not available")


@pytest.fixture(scope="module", autouse=True)
def _jvm():
    ensure_jvm_started()


def _java_report(**fields):
    report = jpype.JClass("group.gnometrading.schemas.OrderExecutionReport")()
    report.encodeClientOid(jpype.JLong(7), jpype.JInt(0))
    enc = report.encoder
    enc.execType(jpype.JClass("group.gnometrading.schemas.ExecType").REJECT)
    for name, value in fields.items():
        getattr(enc, name)(jpype.JLong(value))
    return report


def test_null_fields_read_as_zero():
    nulls = jpype.JClass("group.gnometrading.schemas.OrderExecutionReportDecoder")
    report = _java_report(
        filledQty=nulls.filledQtyNullValue(),
        fillPrice=nulls.fillPriceNullValue(),
        cumulativeQty=nulls.cumulativeQtyNullValue(),
        leavesQty=nulls.leavesQtyNullValue(),
        fee=nulls.feeNullValue(),
        timestampEvent=nulls.timestampEventNullValue(),
        timestampRecv=nulls.timestampRecvNullValue(),
    )

    py = ExecutionReport._from_java(report)

    assert py.exec_type == ExecType.REJECT
    assert (py.filled_qty, py.fill_price, py.cumulative_qty, py.leaves_qty) == (0, 0, 0, 0)
    assert py.fee == 0.0
    assert (py.timestamp_event, py.timestamp_recv) == (0, 0)


def test_present_fields_pass_through():
    report = _java_report(filledQty=2_000_000, fillPrice=450_000_000, fee=1_500_000_000, timestampRecv=123)

    py = ExecutionReport._from_java(report)

    assert py.filled_qty == 2_000_000
    assert py.fill_price == 450_000_000
    assert py.fee == pytest.approx(1.5)
    assert py.timestamp_recv == 123
