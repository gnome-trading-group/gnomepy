"""The journal decoder against hand-built segments, the bytes the orchestrator writes and uploads."""
from __future__ import annotations

import struct

import pytest
import zstandard

from gnomepy.journal import (
    METRIC_COLUMN_TEMPLATE,
    METRIC_ROW_TEMPLATE,
    METRIC_STREAM_TEMPLATE,
    iter_records,
    read_journal_metrics,
    session_journal_uri,
)

_SCHEMA_ID, _VERSION = 1, 1
_MBP10_TEMPLATE = 2
_LONG, _DOUBLE, _INT = 0, 1, 2


def _message(template: int, block: bytes, var_data: bytes) -> bytes:
    header = struct.pack("<HHHH", len(block), template, _SCHEMA_ID, _VERSION)
    return header + block + struct.pack("<H", len(var_data)) + var_data


def _stream(stream_id: int, column_count: int, name: str) -> tuple[int, bytes]:
    return METRIC_STREAM_TEMPLATE, _message(
        METRIC_STREAM_TEMPLATE, struct.pack("<HH", stream_id, column_count), name.encode()
    )


def _column(stream_id: int, index: int, column_type: int, name: str) -> tuple[int, bytes]:
    return METRIC_COLUMN_TEMPLATE, _message(
        METRIC_COLUMN_TEMPLATE, struct.pack("<HHB", stream_id, index, column_type), name.encode()
    )


def _row(stream_id: int, fmt: str, *values) -> tuple[int, bytes]:
    return METRIC_ROW_TEMPLATE, _message(METRIC_ROW_TEMPLATE, struct.pack("<H", stream_id), struct.pack(fmt, *values))


def _market_data() -> tuple[int, bytes]:
    return _MBP10_TEMPLATE, b"\x00" * 40


def _segment(records, first_sequence: int, source: int = 8) -> bytes:
    out = b""
    for offset, (template, payload) in enumerate(records):
        out += struct.pack("<QHHH", first_sequence + offset, source, template, len(payload)) + payload
    # A segment file keeps its mapped size; the zeros after its records are where a reader stops.
    return out + b"\x00" * 64


def _write(directory, index: int, data: bytes, kind: str = "zst") -> None:
    if kind == "zst":
        stream = zstandard.ZstdCompressor().stream_writer(open(directory / f"segment-{index:05d}.zst", "wb"))
        stream.write(data)
        stream.close()
    else:
        (directory / f"segment-{index:05d}.{kind}").write_bytes(data)


_DECLARATIONS = [
    _stream(0, 2, "orders"),
    _column(0, 0, _LONG, "timestamp"),
    _column(0, 1, _DOUBLE, "fair"),
    _stream(1, 1, "levels"),
    _column(1, 0, _INT, "level"),
]


def test_streams_are_rebuilt_across_compressed_segments_indexed_by_sequence(tmp_path):
    _write(tmp_path, 0, _segment([*_DECLARATIONS, _market_data(), _row(0, "<qd", 100, 0.6)], 1))
    _write(tmp_path, 1, _segment([_row(1, "<i", -3), _market_data(), _row(0, "<qd", 200, 0.7)], 8))

    metrics = read_journal_metrics(tmp_path)

    orders = metrics["orders"]
    assert list(orders.columns) == ["timestamp", "fair"]
    assert orders["timestamp"].tolist() == [100, 200]
    assert orders["fair"].tolist() == [0.6, 0.7]
    assert orders.index.tolist() == [7, 10]
    assert metrics["levels"]["level"].tolist() == [-3]
    assert metrics["levels"].index.tolist() == [8]


def test_a_local_run_leaves_sealed_segments_then_the_open_one(tmp_path):
    sealed = _segment(_DECLARATIONS, 1)
    # Sealing trims a segment to its records, so no zeros follow the last one.
    _write(tmp_path, 0, sealed[: len(sealed) - 64], kind="done")
    _write(tmp_path, 1, _segment([_row(1, "<i", 5)], 1 + len(_DECLARATIONS)), kind="bin")

    assert read_journal_metrics(tmp_path)["levels"]["level"].tolist() == [5]


def test_a_missing_segment_is_refused(tmp_path):
    _write(tmp_path, 0, _segment(_DECLARATIONS, 1))
    _write(tmp_path, 2, _segment([_row(1, "<i", 5)], 7))

    with pytest.raises(ValueError, match="missing segments"):
        read_journal_metrics(tmp_path)


def test_a_sequence_gap_is_refused():
    data = _segment([_market_data()], 1)[:-64] + _segment([_market_data()], 3)

    with pytest.raises(ValueError, match="expected 2 but got 3"):
        list(iter_records(data))


def test_a_row_for_an_undeclared_stream_is_refused(tmp_path):
    _write(tmp_path, 0, _segment([_row(0, "<q", 1)], 1))

    with pytest.raises(ValueError, match="never declared"):
        read_journal_metrics(tmp_path)


def test_streams_from_different_sources_keep_their_own_ids(tmp_path):
    strategy = _segment([_stream(0, 1, "strategy"), _column(0, 0, _LONG, "x"), _row(0, "<q", 1)], 1, source=8)
    other = _segment([_stream(0, 1, "oms"), _column(0, 0, _INT, "y"), _row(0, "<i", 2)], 4, source=9)
    _write(tmp_path, 0, strategy[:-64] + other)

    metrics = read_journal_metrics(tmp_path)

    assert metrics["strategy"]["x"].tolist() == [1]
    assert metrics["oms"]["y"].tolist() == [2]


def test_an_empty_journal_directory_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_journal_metrics(tmp_path)


def test_a_sessions_journal_is_under_its_stage_bucket():
    assert session_journal_uri("abc", "dev") == "s3://gnome-journals-dev/abc"
