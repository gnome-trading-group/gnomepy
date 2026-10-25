from __future__ import annotations

import decimal
import io
from typing import Callable, Iterator

import jpype
import numpy as np
import pandas as pd
import zstandard

from gnomepy.java._jvm import ensure_jvm_started
from gnomepy.java.enums import RecordType
from gnomepy.java.sbe import get_message
from gnomepy.java.schemas import StreamRecord, wrap_record
from gnomepy.java.statics import Scales


_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


class DataStore:
    """Records of one stream, from stored bytes or already decoded.

    ``record_type`` names the columns ``to_df`` scales; the bytes themselves are walked message by message, so a file
    written either side of a schema change reads the same.
    """

    def __init__(self, data: bytes, record_type: RecordType):
        if data[:4] == _ZSTD_MAGIC:
            data = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)).read()
        self._data: bytes | None = data
        self._records: list[StreamRecord] | None = None
        self._record_type = record_type

    @classmethod
    def from_bytes(
        cls,
        data: bytes | io.BytesIO | io.RawIOBase | io.BufferedIOBase,
        record_type: RecordType,
    ) -> DataStore:
        if isinstance(data, (bytes, bytearray)):
            raw = bytes(data)
        else:
            raw = data.read()
        return cls(raw, record_type)

    @classmethod
    def from_records(cls, records: list[StreamRecord], record_type: RecordType) -> DataStore:
        instance = object.__new__(cls)
        instance._data = None
        instance._records = records
        instance._record_type = record_type
        return instance

    def _decoded(self) -> list[StreamRecord]:
        if self._records is None:
            ensure_jvm_started()
            stream_entry = jpype.JClass("group.gnometrading.data.StreamEntry")
            java_records = stream_entry.decode(jpype.JArray(jpype.JByte)(self._data or b""))
            self._records = [wrap_record(record) for record in java_records]
        return self._records

    def __iter__(self) -> Iterator[StreamRecord]:
        yield from self._decoded()

    def __len__(self) -> int:
        return len(self._decoded())

    def to_df(
        self,
        price_type: str = "float",
        size_type: str = "float",
        pretty_ts: bool = True,
        tz: str = "UTC",
        replace_nulls: bool = True,
    ) -> pd.DataFrame:
        rows = [s.to_dict() for s in self]
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        msg = get_message(self._record_type)

        if replace_nulls:
            for col, null_val in msg.null_fields.items():
                if col in df.columns:
                    df[col] = df[col].replace(null_val, np.nan)

        _apply_scaling(df, msg.fields_by_type("price"), price_type, Scales.PRICE)
        _apply_scaling(df, msg.fields_by_type("size") + msg.fields_by_type("volume"), size_type, Scales.SIZE)

        if pretty_ts:
            ts_cols = [c for c in msg.fields_by_type("timestamp") if c in df.columns]
            for col in ts_cols:
                df[col] = pd.to_datetime(df[col], unit="ns", utc=True, errors="coerce")
                df[col] = df[col].dt.tz_convert(tz)

        return df

    def replay(self, callback: Callable[[StreamRecord], None]) -> None:
        for record in self:
            callback(record)


def _apply_scaling(
    df: pd.DataFrame,
    cols: list[str],
    scale_type: str,
    scale: int,
) -> None:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return
    if scale_type == "float":
        df[cols] = df[cols].astype(float) / scale
    elif scale_type == "decimal":
        for col in cols:
            df[col] = df[col].apply(lambda v: decimal.Decimal(v) / decimal.Decimal(scale) if pd.notna(v) else v)
