"""The order-path reader against hand-built v3 records, laid out as gnome-schemas' core v3 codecs write them."""
from __future__ import annotations

import struct

import pandas as pd
import pytest

from gnomepy.latency import read_journal_latency

_SEGMENT_HEADER = b"GNJ2" + struct.pack("<H", 2) + b"\x00" * 10
_T0 = 1_700_000_000_000_000_000
_MD_KIND, _EXEC_KIND, _RISK_KIND = 1, 2, 4
_INTENT, _ORDER, _MODIFY, _CANCEL, _REPORT, _MBP10 = 14, 11, 12, 13, 10, 2
_EXEC_NEW, _EXEC_CANCEL, _EXEC_PARTIAL_FILL = 0, 1, 3


def _message(template: int, block: bytes, version: int = 3) -> bytes:
    return struct.pack("<HHHH", len(block), template, 1, version) + block


def _intent(trigger_sequence: int, kind: int, trigger_at: int, origin: int, dispatch: int) -> bytes:
    block = bytearray(100)
    struct.pack_into("<Q", block, 8, dispatch + 50)
    struct.pack_into("<QBQQQ", block, 67, trigger_sequence, kind, trigger_at, origin, dispatch)
    return _message(_INTENT, bytes(block))


def _oid(counter: int) -> bytes:
    return struct.pack("<QQ", counter, 7)


def _order(oid: int, intent_sequence: int, trigger_sequence: int, kind: int, trigger_at: int, origin: int) -> bytes:
    block = bytearray(83)
    struct.pack_into("<HIQ16s", block, 0, 4, 42, 0, _oid(oid))
    struct.pack_into("<QQBQQ", block, 50, intent_sequence, trigger_sequence, kind, trigger_at, origin)
    return _message(_ORDER, bytes(block))


def _cancel(oid: int, trigger_sequence: int, kind: int, trigger_at: int) -> bytes:
    block = bytearray(63)
    struct.pack_into("<HIQ16s", block, 0, 4, 42, 0, _oid(oid))
    struct.pack_into("<QQBQQ", block, 30, 0, trigger_sequence, kind, trigger_at, 0)
    return _message(_CANCEL, bytes(block))


def _report(oid: int, exec_type: int) -> bytes:
    block = bytearray(149)
    struct.pack_into("<16sB", block, 72, _oid(oid), exec_type)
    return _message(_REPORT, bytes(block))


def _write(tmp_path, records: list[tuple[int, int, int, bytes]]) -> str:
    """Records as (arrival offset from T0, source, template, payload), sequenced from 1."""
    out = _SEGMENT_HEADER
    for sequence, (arrival, source, template, payload) in enumerate(records, start=1):
        out += struct.pack("<QQHHH", sequence, _T0 + arrival, source, template, len(payload)) + payload
    (tmp_path / "segment-00000.done").write_bytes(out)
    return str(tmp_path)


def test_an_order_is_timed_from_its_trigger_through_its_intent_to_its_ack(tmp_path):
    journal = _write(tmp_path, [
        (1_000, 7, _MBP10, _message(_MBP10, b"\x00" * 40)),
        (1_500, 1, _INTENT, _intent(1, _MD_KIND, _T0 + 1_000, _T0 + 200, _T0 + 1_100)),
        (1_800, 3, _ORDER, _order(9, 2, 1, _MD_KIND, _T0 + 1_000, _T0 + 200)),
        (90_000, 4, _REPORT, _report(9, _EXEC_NEW)),
    ])

    frame = read_journal_latency(journal)

    assert list(frame.index) == [3]
    row = frame.loc[3]
    assert row["action"] == "NEW"
    assert row["trigger_kind"] == "MARKET_DATA"
    assert row["exchange_id"] == 4
    assert row["intent_sequence"] == 2
    assert row["trigger_transit"] == 800
    assert row["strat_queue"] == 100
    assert row["strat_decision"] == 400
    assert row["oms_process"] == 300
    assert row["tick_to_order"] == 800
    assert row["order_to_ack"] == 88_200


def test_a_cancel_is_answered_by_its_own_report_not_the_orders_ack(tmp_path):
    journal = _write(tmp_path, [
        (1_000, 3, _ORDER, _order(9, 0, 0, _RISK_KIND, 0, 0)),
        (2_000, 3, _CANCEL, _cancel(9, 0, _RISK_KIND, 0)),
        (5_000, 4, _REPORT, _report(9, _EXEC_NEW)),
        (9_000, 4, _REPORT, _report(9, _EXEC_CANCEL)),
    ])

    frame = read_journal_latency(journal)

    assert frame.loc[1, "order_to_ack"] == 4_000
    assert frame.loc[2, "order_to_ack"] == 7_000
    assert frame.loc[2, "trigger_kind"] == "RISK"
    assert pd.isna(frame.loc[2, "tick_to_order"])
    assert pd.isna(frame.loc[2, "oms_process"])


def test_an_order_reacting_to_a_report_is_processed_from_the_reports_arrival(tmp_path):
    journal = _write(tmp_path, [
        (1_000, 4, _REPORT, _report(3, _EXEC_NEW)),
        (1_400, 3, _ORDER, _order(4, 0, 1, _EXEC_KIND, _T0 + 1_000, _T0 + 600)),
    ])

    row = read_journal_latency(journal).loc[2]

    assert row["trigger_kind"] == "EXEC_REPORT"
    assert pd.isna(row["intent_sequence"])
    assert row["oms_process"] == 400


def test_records_from_before_v3_are_skipped(tmp_path):
    old = struct.pack("<HHHH", 50, _ORDER, 1, 2) + b"\x00" * 50
    journal = _write(tmp_path, [(1_000, 3, _ORDER, old)])

    frame = read_journal_latency(journal)

    assert frame.empty
    assert "tick_to_order" in frame.columns


@pytest.mark.parametrize("missing", ["a journal with no segments"])
def test_no_segments_is_an_error(tmp_path, missing):
    with pytest.raises(FileNotFoundError):
        read_journal_latency(str(tmp_path))


def _modify(oid: int) -> bytes:
    block = bytearray(82)
    struct.pack_into("<HIQ16s", block, 0, 4, 42, 0, _oid(oid))
    struct.pack_into("<QQBQQ", block, 49, 0, 0, _RISK_KIND, 0, 0)
    return _message(_MODIFY, bytes(block))


def test_a_fill_of_the_resting_order_does_not_answer_its_modify(tmp_path):
    journal = _write(tmp_path, [
        (1_000, 3, _ORDER, _order(9, 0, 0, _RISK_KIND, 0, 0)),
        (2_000, 4, _REPORT, _report(9, _EXEC_NEW)),
        (3_000, 3, _MODIFY, _modify(9)),
        (4_000, 4, _REPORT, _report(9, _EXEC_PARTIAL_FILL)),
        (7_000, 4, _REPORT, _report(9, _EXEC_NEW)),
    ])

    frame = read_journal_latency(journal)

    assert frame.loc[3, "action"] == "MODIFY"
    assert frame.loc[3, "order_to_ack"] == 4_000


_TIMING_SOURCE, _TIMING_TEMPLATE = 10, 212
_NEW_CLASS, _CANCEL_CLASS = 3, 0
_ANSWERED, _NOT_SENT = 0, 4


def _timing(
    counter: int,
    request_class: int,
    stamps: tuple[int, int, int, int, int],
    outcome: int = _ANSWERED,
    code: int = 201,
    venue_time: int = 0,
    processing_micros: int = -1,
    slack: int = 0,
) -> bytes:
    ingress, taken, decided, wired, answered = stamps
    block = struct.pack(
        "<iqqBBqqqqqqiqi", 5, 42, counter, request_class, outcome, ingress, taken, decided, wired, answered, slack,
        code, venue_time, processing_micros,
    )
    return struct.pack("<HHHH", len(block), _TIMING_TEMPLATE, 3, 0) + block


def test_an_orders_gateway_timing_splits_its_path_past_the_session(tmp_path):
    journal = _write(tmp_path, [
        (1_800, 3, _ORDER, _order(9, 0, 1, _MD_KIND, _T0 + 1_000, _T0 + 200)),
        (60_000, 4, _REPORT, _report(9, _EXEC_NEW)),
        (61_000, _TIMING_SOURCE, _TIMING_TEMPLATE, _timing(
            9, _NEW_CLASS, (_T0 + 2_300, _T0 + 2_400, _T0 + 2_600, _T0 + 3_000, _T0 + 50_000),
            processing_micros=30, slack=40,
        )),
    ])

    row = read_journal_latency(journal).loc[1]

    assert row["gw_outcome"] == "ANSWERED"
    assert row["gw_attempts"] == 1
    assert row["venue_code"] == 201
    assert row["gw_transit"] == 500
    assert row["gw_queue"] == 100
    assert row["gw_budget_wait"] == 200
    assert row["gw_sign_send"] == 400
    assert row["venue_answer"] == 47_000
    assert row["venue_processing"] == 30_000
    assert row["venue_network"] == 17_000
    assert row["answer_slack"] == 40
    assert row["tick_to_trade"] == 2_800, "from the platform's receipt of the trigger to the wire"
    assert pd.isna(row["venue_time"])


def test_a_cancel_sent_again_counts_each_attempt_and_keeps_the_firsts_stamps(tmp_path):
    first = (_T0 + 2_100, _T0 + 2_200, _T0 + 2_300, _T0 + 2_400, 0)
    second = (_T0 + 2_100, _T0 + 2_200, _T0 + 9_000, _T0 + 9_100, _T0 + 12_000)
    journal = _write(tmp_path, [
        (1_000, 3, _ORDER, _order(9, 0, 0, _RISK_KIND, 0, 0)),
        (2_000, 3, _CANCEL, _cancel(9, 0, _RISK_KIND, 0)),
        (5_000, _TIMING_SOURCE, _TIMING_TEMPLATE, _timing(9, _CANCEL_CLASS, first, outcome=3, code=0)),
        (13_000, _TIMING_SOURCE, _TIMING_TEMPLATE, _timing(9, _CANCEL_CLASS, second)),
    ])

    frame = read_journal_latency(journal)

    assert pd.isna(frame.loc[1, "gw_outcome"]), "the order itself was never timed"
    assert frame.loc[1, "gw_attempts"] == 0
    cancel = frame.loc[2]
    assert cancel["gw_attempts"] == 2
    assert cancel["gw_outcome"] == "NO_ANSWER"
    assert pd.isna(cancel["venue_answered"])
    assert pd.isna(cancel["venue_code"])
    assert cancel["wire_sent"] == _T0 + 2_400


def test_a_refused_order_is_timed_not_sent_and_a_timing_for_no_order_is_ignored(tmp_path):
    journal = _write(tmp_path, [
        (1_000, 3, _ORDER, _order(9, 0, 0, _RISK_KIND, 0, 0)),
        (2_000, _TIMING_SOURCE, _TIMING_TEMPLATE, _timing(
            9, _NEW_CLASS, (_T0 + 1_100, _T0 + 1_200, _T0 + 1_700, 0, 0), outcome=_NOT_SENT, code=0,
        )),
        (3_000, _TIMING_SOURCE, _TIMING_TEMPLATE, _timing(77, _NEW_CLASS, (1, 2, 3, 4, 5))),
    ])

    frame = read_journal_latency(journal)

    assert len(frame) == 1
    assert frame.loc[1, "gw_outcome"] == "NOT_SENT"
    assert frame.loc[1, "gw_budget_wait"] == 500
    assert pd.isna(frame.loc[1, "wire_sent"])
    assert pd.isna(frame.loc[1, "venue_answer"])


def test_stamps_keep_every_nanosecond_when_other_orders_lack_them(tmp_path):
    journal = _write(tmp_path, [
        (1_000, 3, _ORDER, _order(8, 0, 0, _RISK_KIND, 0, 0)),
        (1_800, 3, _ORDER, _order(9, 0, 1, _MD_KIND, _T0 + 1_001, _T0 + 203)),
    ])

    frame = read_journal_latency(journal)

    assert pd.isna(frame.loc[1, "trigger_origin"])
    assert frame.loc[2, "trigger_origin"] == _T0 + 203
    assert frame.loc[2, "trigger_transit"] == 798
    assert frame["exchange_id"].dtype == "int64"
