"""Reads the journal a live or paper session ships to S3 as it runs, and the strategy metrics recorded in it.

A session's journal is ``s3://gnome-journals-<stage>/<session_id>/segment-NNNNN.zst``: zstd-compressed segments
holding, in one global order, every message the session saw and sent. Each record is a 14-byte header
(global sequence u64, source id u16, template id u16, payload length u16, little-endian) and an SBE payload.

Metric streams are declared at runtime, so the journal carries their declarations ahead of their rows: a
``MetricStream`` record names a stream, a ``MetricColumn`` record per column names it and gives its type, and each
``MetricRow`` holds one row's values packed in declaration order. Decoding needs no JVM.
"""

from __future__ import annotations

import io
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import zstandard

from gnomepy._fs import fs_list_files, fs_read_bytes, resolve_fs

_RECORD_HEADER = struct.Struct("<QHHH")
_MESSAGE_HEADER = struct.Struct("<HHHH")
_U16 = struct.Struct("<H")
_STREAM_BLOCK = struct.Struct("<HH")
_COLUMN_BLOCK = struct.Struct("<HHB")

# Template ids of the metric messages in gnome-schemas' core.xml.
METRIC_ROW_TEMPLATE = 15
METRIC_STREAM_TEMPLATE = 16
METRIC_COLUMN_TEMPLATE = 17

# MetricColumnType values, as numpy little-endian dtypes.
_COLUMN_DTYPES = {0: "<i8", 1: "<f8", 2: "<i4"}

_SEGMENT_NAME = re.compile(r"segment-(\d+)\.(zst|done|bin)$")


def session_journal_uri(session_id: str, stage: str = "prod") -> str:
    """Where a session's journal segments are uploaded."""
    return f"s3://gnome-journals-{stage}/{session_id}"


def load_session_metrics(session_id: str, stage: str = "prod") -> dict[str, pd.DataFrame]:
    """Every metric stream a live or paper session recorded, by stream name.

    Each DataFrame has the stream's columns in declaration order, indexed by ``sequence``: the record's place in the
    session's journal, which orders it exactly against the market data, intents and execution reports around it.

    Downloads the whole journal, market data included, so a long session takes a while.
    """
    return read_journal_metrics(session_journal_uri(session_id, stage))


def read_journal_metrics(journal: str | Path) -> dict[str, pd.DataFrame]:
    """The metric streams in a journal directory: an S3 prefix of uploaded segments, or a local one.

    Uploaded segments are ``segment-NNNNN.zst``. A local run leaves them uncompressed: ``segment-NNNNN.done`` once
    finished, ``segment-NNNNN.bin`` while open or if the session crashed.
    """
    decoder = _MetricDecoder()
    previous = 0
    for segment in read_segments(journal):
        for sequence, source_id, template_id, payload in iter_records(segment, previous):
            decoder.feed(sequence, source_id, template_id, payload)
            previous = sequence
    return decoder.frames()


def read_segments(journal: str | Path) -> Iterator[bytes]:
    """Each segment's bytes, in journal order."""
    fs, base = resolve_fs(str(journal))
    segments = []
    for path in fs_list_files(fs, base):
        match = _SEGMENT_NAME.search(path)
        if match:
            segments.append((int(match.group(1)), match.group(2), path))
    if not segments:
        raise FileNotFoundError(f"No journal segments under {journal}")
    segments.sort()
    indexes = [index for index, _, _ in segments]
    if indexes != list(range(len(segments))):
        raise ValueError(f"Journal under {journal} is missing segments: found {indexes}")
    for _, kind, path in segments:
        data = fs_read_bytes(fs, path)
        if kind == "zst":
            # Segments are compressed as a stream, so their frames carry no content size to decompress against.
            data = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)).read()
        yield data


def iter_records(segment: bytes, previous_sequence: int = 0) -> Iterator[tuple[int, int, int, memoryview]]:
    """Each record in a segment as (global sequence, source id, template id, payload).

    Raises:
        ValueError: if a record doesn't follow the one before it, so a journal with a missing segment never reads as
            whole.
    """
    view = memoryview(segment)
    offset = 0
    expected = previous_sequence + 1
    end = len(view)
    while offset + _RECORD_HEADER.size <= end:
        sequence, source_id, template_id, length = _RECORD_HEADER.unpack_from(view, offset)
        if sequence == 0:
            return
        if sequence != expected:
            raise ValueError(f"Journal sequence gap: expected {expected} but got {sequence}")
        offset += _RECORD_HEADER.size
        if offset + length > end:
            return
        yield sequence, source_id, template_id, view[offset : offset + length]
        offset += length
        expected += 1


def _var_data(payload: memoryview) -> memoryview:
    """A message's one var-data field, found past its block so the block can grow without breaking readers."""
    block_length = _MESSAGE_HEADER.unpack_from(payload, 0)[0]
    start = _MESSAGE_HEADER.size + block_length
    (length,) = _U16.unpack_from(payload, start)
    return payload[start + _U16.size : start + _U16.size + length]


@dataclass
class _Stream:
    name: str
    column_count: int
    columns: dict[int, tuple[str, int]] = field(default_factory=dict)
    rows: list[bytes] = field(default_factory=list)
    sequences: list[int] = field(default_factory=list)

    def frame(self) -> pd.DataFrame:
        if len(self.columns) != self.column_count:
            raise ValueError(
                f"Metric stream {self.name!r} declared {self.column_count} columns but the journal holds "
                f"{len(self.columns)}"
            )
        dtype = np.dtype([
            (name, _COLUMN_DTYPES[column_type]) for name, column_type in (self.columns[i] for i in range(self.column_count))
        ])
        values = np.frombuffer(b"".join(self.rows), dtype=dtype)
        frame = pd.DataFrame({name: values[name] for name in dtype.names})
        frame.index = pd.Index(np.array(self.sequences, dtype=np.int64), name="sequence")
        return frame


class _MetricDecoder:
    """Rebuilds metric streams from their declarations and rows. Stream ids are per source, so keys are both."""

    def __init__(self):
        self._streams: dict[tuple[int, int], _Stream] = {}

    def feed(self, sequence: int, source_id: int, template_id: int, payload: memoryview) -> None:
        if template_id == METRIC_ROW_TEMPLATE:
            (stream_id,) = _U16.unpack_from(payload, _MESSAGE_HEADER.size)
            stream = self._stream(source_id, stream_id)
            stream.rows.append(bytes(_var_data(payload)))
            stream.sequences.append(sequence)
        elif template_id == METRIC_STREAM_TEMPLATE:
            stream_id, column_count = _STREAM_BLOCK.unpack_from(payload, _MESSAGE_HEADER.size)
            name = bytes(_var_data(payload)).decode("ascii")
            self._streams[(source_id, stream_id)] = _Stream(name, column_count)
        elif template_id == METRIC_COLUMN_TEMPLATE:
            stream_id, column_index, column_type = _COLUMN_BLOCK.unpack_from(payload, _MESSAGE_HEADER.size)
            if column_type not in _COLUMN_DTYPES:
                raise ValueError(f"Unknown metric column type {column_type}")
            name = bytes(_var_data(payload)).decode("ascii")
            self._stream(source_id, stream_id).columns[column_index] = (name, column_type)

    def _stream(self, source_id: int, stream_id: int) -> _Stream:
        stream = self._streams.get((source_id, stream_id))
        if stream is None:
            raise ValueError(f"Metric stream {stream_id} from source {source_id} was never declared")
        return stream

    def frames(self) -> dict[str, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        for stream in self._streams.values():
            if stream.name in frames:
                raise ValueError(f"Two sources in the journal declare a metric stream named {stream.name!r}")
            frames[stream.name] = stream.frame()
        return frames
