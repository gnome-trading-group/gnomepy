"""Where a session's time went between an event and the orders it caused, read from the session's journal.

Since core schema v3 every intent carries the event the strategy was handling when it published it (its journal
sequence, kind, when it reached the session and when the platform received it) and when handling began; every order,
modify and cancel the OMS sends carries the intent it serves and that trigger, or the execution report it reacts to.
Journal sequences join them, and the journal's own arrival stamps say when each record reached its ring, so each
order's path can be timed from the journal alone, on the session host's clock.

Records from before v3 carry none of this and are skipped. The offline path ends where the OMS publishes the order:
the hand-off to the order gateway isn't journaled (the session's heartbeat and ``_latency`` metric stream measure it
live).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy.journal import iter_records, read_segments, session_journal_uri

# Journal sources (gnome-orchestrator's JournalSource).
STRATEGY_INTENT_SOURCE = 1
OMS_ORDER_SOURCE = 3
OMS_EXEC_REPORT_SOURCE = 4
GATEWAY_TIMING_SOURCE = 10

# Template ids in gnome-schemas' core.xml.
_EXEC_REPORT_TEMPLATE = 10
_ORDER_TEMPLATE = 11
_MODIFY_TEMPLATE = 12
_CANCEL_TEMPLATE = 13
_INTENT_TEMPLATE = 14

_HEADER = struct.Struct("<HHHH")
_CORE_V3 = 3

TRIGGER_KINDS = {0: "NONE", 1: "MARKET_DATA", 2: "EXEC_REPORT", 3: "STREAM", 4: "RISK", 5: "OTHER"}
_EXEC_CANCEL, _EXEC_FILL, _EXEC_REJECT, _EXEC_CANCEL_REJECT, _EXEC_EXPIRE = 1, 2, 4, 5, 6
# Reports that answer a cancel: its own answer, or the order's end.
_ANSWERS_CANCEL = {_EXEC_CANCEL, _EXEC_CANCEL_REJECT, _EXEC_FILL, _EXEC_EXPIRE, _EXEC_REJECT}
_EXEC_NEW, _EXEC_PARTIAL_FILL = 0, 3
# Reports that answer a modify: the venue taking it (NEW), refusing it, or the order ending. A fill of the resting order
# can arrive before the venue answers the modify, and says nothing about it.
_ANSWERS_MODIFY = {_EXEC_NEW, _EXEC_CANCEL, _EXEC_CANCEL_REJECT, _EXEC_REJECT, _EXEC_EXPIRE}
_ANSWERS = {"CANCEL": _ANSWERS_CANCEL, "MODIFY": _ANSWERS_MODIFY}

# Field offsets in each message's v3 block (from the generated SBE codecs).
_INTENT = struct.Struct("<8xQ")  # timestampSend at 8
_INTENT_TRIGGER = struct.Struct("<QBQQQ")  # at 67: triggerSequence, triggerKind, triggerTimestamp, origin, dispatch
_INTENT_TRIGGER_OFFSET = 67
_ORDER_PREFIX = struct.Struct("<HIQ16s")  # exchangeId, securityId, timestampSend, clientOid
_CAUSE = struct.Struct("<QQBQQ")  # intentSequence, triggerSequence, triggerKind, triggerTimestamp, origin
_CAUSE_OFFSET = {_ORDER_TEMPLATE: 50, _MODIFY_TEMPLATE: 49, _CANCEL_TEMPLATE: 30}
_ACTIONS = {_ORDER_TEMPLATE: "NEW", _MODIFY_TEMPLATE: "MODIFY", _CANCEL_TEMPLATE: "CANCEL"}
_REPORT_CLIENT_OID = struct.Struct("<16sB")  # clientOid at 72, then execType
_REPORT_CLIENT_OID_OFFSET = 72

# A gateway's OrderTiming (gateway schema 212): venueAccountId, sessionSeq, clientOidCounter, requestClass, outcome,
# ingress, taken, decided, wireSent, answered, answerSlack, venueCode, venueTs, venueProcessingMicros.
_ORDER_TIMING_TEMPLATE = 212
_TIMING = struct.Struct("<iqqBBqqqqqqiqi")
_TIMING_ACTIONS = {0: "CANCEL", 2: "MODIFY", 3: "NEW"}
_TIMING_OUTCOMES = {0: "ANSWERED", 1: "REJECTED", 2: "TRANSPORT_ERROR", 3: "NO_ANSWER", 4: "NOT_SENT"}
_TIMING_STAMPS = ("gw_ingress", "gw_taken", "gw_decided", "wire_sent", "venue_answered")

_COLUMNS = [
    "action", "exchange_id", "security_id", "client_oid", "trigger_kind", "trigger_sequence", "intent_sequence",
    "trigger_origin", "trigger_arrival", "dispatch", "intent_arrival", "order_arrival", "ack_arrival",
    "gw_outcome", "gw_attempts", "venue_code", *_TIMING_STAMPS, "answer_slack", "venue_time", "venue_processing",
    "trigger_transit", "strat_queue", "strat_decision", "oms_process", "tick_to_order", "order_to_ack",
    "gw_transit", "gw_queue", "gw_budget_wait", "gw_sign_send", "venue_answer", "venue_network", "tick_to_trade",
]
_STAMPED = 24  # the columns read from the journal, before the derived ones


def load_session_latency(session_id: str, stage: str = "prod") -> pd.DataFrame:
    """One row per order, modify and cancel a live or paper session's OMS sent, with every stamp along its path.

    Downloads the whole journal, market data included, so a long session takes a while. See
    :func:`read_journal_latency` for the columns.
    """
    return read_journal_latency(session_journal_uri(session_id, stage))


def read_journal_latency(journal: str | Path) -> pd.DataFrame:
    """The order path's stamps in a journal directory (an S3 prefix of uploaded segments, or a local one).

    Indexed by the order's journal ``sequence``. Times are epoch nanoseconds on the session host's clock, and
    durations nanoseconds, all nullable ``Int64`` (epoch nanoseconds don't fit a float exactly); a stamp the order
    doesn't have is ``<NA>``:

    - ``trigger_origin``: when the platform received the trigger (a market-data record's shard receipt, a report's
      gateway receipt); ``trigger_arrival``: when it reached the session's ring.
    - ``dispatch``: when the strategy began handling it; ``intent_arrival``: when its intent reached the OMS's ring;
      ``order_arrival``: when the OMS published the order; ``ack_arrival``: when the answer reached the session (an
      order's first report, a cancel's own answer or its order's end).
    - ``trigger_transit`` = arrival - origin (crosses hosts: carries both clocks' error), ``strat_queue`` = dispatch -
      arrival, ``strat_decision`` = intent - dispatch, ``oms_process`` = order - intent (or - the report's arrival, for
      an order the OMS sent reacting to a report), ``tick_to_order`` = order - trigger arrival, ``order_to_ack`` = ack -
      order.

    An order the OMS sent on its own (``RISK``: a kill or halt sweep, shutdown) has no trigger.

    From the order's gateway (its ``OrderTiming``, journaled as source 10 by sessions on an orchestrator that has it;
    absent for older sessions, and for a request its gateway refused before its lane): times on the gateway host's
    clock, epoch nanoseconds.

    - ``gw_outcome`` (ANSWERED, REJECTED, TRANSPORT_ERROR, NO_ANSWER, NOT_SENT), ``venue_code`` (its HTTP status),
      ``gw_attempts`` (requests the gateway sent for it: a resent submit or a cancel sent again counts each; the stamps
      are the first's).
    - ``gw_ingress`` (read off the session's image), ``gw_taken`` (off the lane's ring), ``gw_decided`` (the budget let
      it go, or refused it), ``wire_sent`` (written to the venue's socket), ``venue_answered`` (its answer found);
      ``answer_slack``: how long before ``venue_answered`` the lane last looked, the most that stamp can be late.
    - ``venue_time``: the venue's own time for it (Kalshi's ``ts_ms``, ms); ``venue_processing``: how long the venue
      says it took, nanoseconds (Polymarket US's ``x-pm-server-latency``, Polymarket Intl's Cloudflare edge + worker).
    - ``gw_transit`` = gateway ingress - order (crosses hosts), ``gw_queue`` = taken - ingress, ``gw_budget_wait`` =
      decided - taken, ``gw_sign_send`` = wire - decided, ``venue_answer`` = answered - wire, ``venue_network`` =
      venue_answer - venue_processing, ``tick_to_trade`` = wire - trigger origin (crosses hosts).
    """
    intents: dict[int, tuple[int | None, int, int]] = {}
    rows: list[dict] = []
    sequences: list[int] = []
    waiting: dict[bytes, list[int]] = {}
    requests: dict[tuple[int, str], list[int]] = {}
    previous = 0
    for segment in read_segments(journal):
        for sequence, arrival, source_id, template_id, payload in iter_records(segment, previous):
            previous = sequence
            if source_id == GATEWAY_TIMING_SOURCE and template_id == _ORDER_TIMING_TEMPLATE:
                _timing(payload, rows, requests)
                continue
            if not _is_v3(payload):
                continue
            if source_id == STRATEGY_INTENT_SOURCE and template_id == _INTENT_TEMPLATE:
                intents[sequence] = _intent(payload, arrival)
            elif source_id == OMS_ORDER_SOURCE and template_id in _CAUSE_OFFSET:
                row = _order(payload, template_id, arrival, intents)
                waiting.setdefault(row["client_oid"], []).append(len(rows))
                counter = int.from_bytes(row["client_oid"][:8], "little")
                requests.setdefault((counter, row["action"]), []).append(len(rows))
                rows.append(row)
                sequences.append(sequence)
            elif source_id == OMS_EXEC_REPORT_SOURCE and template_id == _EXEC_REPORT_TEMPLATE:
                _answer(payload, arrival, rows, waiting)
    # As objects: pandas would read a stamp column with any missing value as float64, losing epoch nanoseconds.
    frame = pd.DataFrame(rows, columns=_COLUMNS[:_STAMPED], dtype=object)
    _derive(frame)
    frame = frame.infer_objects()
    frame.index = pd.Index(np.array(sequences, dtype=np.int64), name="sequence")
    frame["client_oid"] = frame["client_oid"].map(lambda oid: oid.hex() if isinstance(oid, bytes) else oid)
    return frame[_COLUMNS]


def _is_v3(payload: memoryview) -> bool:
    return len(payload) >= _HEADER.size and _HEADER.unpack_from(payload, 0)[3] >= _CORE_V3


def _intent(payload: memoryview, arrival: int | None) -> tuple[int | None, int, int]:
    (sent,) = _INTENT.unpack_from(payload, _HEADER.size)
    _, _, _, _, dispatch = _INTENT_TRIGGER.unpack_from(payload, _HEADER.size + _INTENT_TRIGGER_OFFSET)
    return arrival, dispatch, sent


def _order(payload: memoryview, template_id: int, arrival: int | None, intents: dict) -> dict:
    exchange_id, security_id, _, client_oid = _ORDER_PREFIX.unpack_from(payload, _HEADER.size)
    intent_sequence, trigger_sequence, kind, trigger_at, origin = _CAUSE.unpack_from(
        payload, _HEADER.size + _CAUSE_OFFSET[template_id]
    )
    intent = intents.get(intent_sequence) if intent_sequence else None
    return {
        "action": _ACTIONS[template_id],
        "exchange_id": exchange_id,
        "security_id": security_id,
        "client_oid": client_oid,
        "trigger_kind": TRIGGER_KINDS.get(kind, str(kind)),
        "trigger_sequence": trigger_sequence or None,
        "intent_sequence": intent_sequence or None,
        "trigger_origin": origin or None,
        "trigger_arrival": trigger_at or None,
        "dispatch": (intent[1] or None) if intent else None,
        "intent_arrival": intent[0] if intent else None,
        "order_arrival": arrival,
        "ack_arrival": None,
        "gw_attempts": 0,
    }


def _timing(payload: memoryview, rows: list[dict], requests: dict[tuple[int, str], list[int]]) -> None:
    """Joins a gateway's timing to the earliest of its order's requests of that action not yet timed."""
    (
        _account, _session, counter, request_class, outcome, ingress, taken, decided, wired, answered, slack, code,
        venue_time, processing_micros,
    ) = _TIMING.unpack_from(payload, _HEADER.size)
    action = _TIMING_ACTIONS.get(request_class)
    indexes = requests.get((counter, action)) if action else None
    if not indexes:
        return
    untimed = [index for index in indexes if rows[index]["gw_attempts"] == 0]
    row = rows[untimed[0] if untimed else indexes[-1]]
    row["gw_attempts"] += 1
    if row["gw_attempts"] > 1:
        return
    row.update({
        "gw_outcome": _TIMING_OUTCOMES.get(outcome, str(outcome)),
        "venue_code": code or None,
        "gw_ingress": ingress or None,
        "gw_taken": taken or None,
        "gw_decided": decided or None,
        "wire_sent": wired or None,
        "venue_answered": answered or None,
        "answer_slack": slack if answered else None,
        "venue_time": venue_time or None,
        "venue_processing": processing_micros * 1_000 if processing_micros >= 0 else None,
    })


def _answer(payload: memoryview, arrival: int | None, rows: list[dict], waiting: dict[bytes, list[int]]) -> None:
    client_oid, exec_type = _REPORT_CLIENT_OID.unpack_from(payload, _HEADER.size + _REPORT_CLIENT_OID_OFFSET)
    pending = waiting.get(client_oid)
    if not pending:
        return
    for position, index in enumerate(pending):
        answers = _ANSWERS.get(rows[index]["action"])
        if answers is None or exec_type in answers:
            rows[index]["ack_arrival"] = arrival
            del pending[position]
            return


def _derive(frame: pd.DataFrame) -> None:
    # Epoch nanoseconds are past float64's exact integers, so stamps stay nullable int64.
    for column in (
        "trigger_sequence", "intent_sequence", *_COLUMNS[7:13], "venue_code", *_TIMING_STAMPS, "answer_slack",
        "venue_time", "venue_processing",
    ):
        frame[column] = pd.array(
            [None if value is None or pd.isna(value) else int(value) for value in frame[column]], dtype="Int64"
        )
    frame["trigger_transit"] = frame["trigger_arrival"] - frame["trigger_origin"]
    frame["strat_queue"] = frame["dispatch"] - frame["trigger_arrival"]
    frame["strat_decision"] = frame["intent_arrival"] - frame["dispatch"]
    handled_from = frame["intent_arrival"].where(frame["intent_sequence"].notna(), frame["trigger_arrival"])
    frame["oms_process"] = frame["order_arrival"] - handled_from
    frame["tick_to_order"] = frame["order_arrival"] - frame["trigger_arrival"]
    frame["order_to_ack"] = frame["ack_arrival"] - frame["order_arrival"]
    frame["gw_attempts"] = frame["gw_attempts"].astype("int64")
    frame["gw_transit"] = frame["gw_ingress"] - frame["order_arrival"]
    frame["gw_queue"] = frame["gw_taken"] - frame["gw_ingress"]
    frame["gw_budget_wait"] = frame["gw_decided"] - frame["gw_taken"]
    frame["gw_sign_send"] = frame["wire_sent"] - frame["gw_decided"]
    frame["venue_answer"] = frame["venue_answered"] - frame["wire_sent"]
    frame["venue_network"] = frame["venue_answer"] - frame["venue_processing"]
    frame["tick_to_trade"] = frame["wire_sent"] - frame["trigger_origin"]
