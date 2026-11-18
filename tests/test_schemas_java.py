"""Book records read through the Python wrappers — requires the gnome-backtest uber JAR; skipped without it."""
from __future__ import annotations

import pytest

from gnomepy.java._classpath import discover_classpath
from gnomepy.java._jvm import ensure_jvm_started
from gnomepy.java.enums import Side
from gnomepy.java.schemas import (Bbo1sSchema, Mbp1Schema, Mbp10Schema, MboSchema, StreamRecord, TradesSchema,
                                   wrap_record)


def _jar_available() -> bool:
    try:
        return len(discover_classpath("gnome-backtest")) > 0
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not _jar_available(), reason="gnome-backtest JAR not available")


@pytest.fixture(scope="module", autouse=True)
def _jvm():
    ensure_jvm_started()


def _levels(n: int) -> dict:
    out = {}
    for i in range(n):
        out |= {f"bid_price_{i}": 100 - i, f"ask_price_{i}": 101 + i, f"bid_size_{i}": 10 + i,
                f"ask_size_{i}": 20 + i, f"bid_count_{i}": 1 + i, f"ask_count_{i}": 2 + i}
    return out


def test_every_level_of_an_mbp10_record_reads_back():
    record = wrap_record(Mbp10Schema(stream_id=7, timestamp_event=5, price=99, size=3, action="Trade",
                                     **_levels(10)).raw)
    assert [(record.bid_price(i), record.bid_size(i), record.bid_count(i)) for i in range(10)] == \
        [(100 - i, 10 + i, 1 + i) for i in range(10)]
    assert [(record.ask_price(i), record.ask_size(i), record.ask_count(i)) for i in range(10)] == \
        [(101 + i, 20 + i, 2 + i) for i in range(10)]
    assert (record.timestamp_event, record.price, record.size, record.action) == (5, 99, 3, "Trade")


def test_an_mbp1_record_reads_its_one_level_beside_an_mbp10_one():
    deep = wrap_record(Mbp10Schema(stream_id=1, **_levels(10)).raw)
    top = wrap_record(Mbp1Schema(stream_id=2, **_levels(1)).raw)
    assert (top.bid_price(0), top.ask_size(0), deep.bid_price(9)) == (100, 20, 91)
    with pytest.raises(IndexError):
        top.bid_price(1)


def test_a_wrapper_reads_its_own_record_not_another_of_the_same_type():
    first = wrap_record(Mbp10Schema(stream_id=1, **_levels(1)).raw)
    second = wrap_record(Mbp10Schema(stream_id=2, bid_price_0=55, bid_size_0=1).raw)
    assert (first.bid_price(0), second.bid_price(0), first.bid_price(0)) == (100, 55, 100)


@pytest.mark.parametrize("make", [Mbp10Schema, MboSchema, TradesSchema])
def test_each_records_side_and_action_read_back_record_by_record(make):
    seen = [(wrap_record(make(stream_id=1, action=action, side=side).raw).side,
             wrap_record(make(stream_id=1, action=action, side=side).raw).action)
            for action, side in (("Trade", "Bid"), ("Add", "Ask"), ("Cancel", "None"), ("Trade", "Ask"))]
    assert seen == [(Side.BID, "Trade"), (Side.ASK, "Add"), (Side.NONE, "Cancel"), (Side.ASK, "Trade")]


def test_a_bbo_records_side_reads_back():
    assert [wrap_record(Bbo1sSchema(stream_id=1, side=side).raw).side for side in ("Ask", "Bid")] == [Side.ASK, Side.BID]


def test_each_record_class_is_wrapped_as_its_own_type():
    records = [Mbp10Schema(stream_id=1), Mbp1Schema(stream_id=1), TradesSchema(stream_id=1), Mbp10Schema(stream_id=2)]
    assert [type(wrap_record(r.raw)) for r in records] == [Mbp10Schema, Mbp1Schema, TradesSchema, Mbp10Schema]
    assert all(isinstance(wrap_record(r.raw), StreamRecord) for r in records)
