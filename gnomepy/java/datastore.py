from __future__ import annotations

import decimal
import io
from typing import Callable, Iterator

import jpype
import numpy as np
import pandas as pd
import zstandard

from gnomepy.java._jvm import ensure_jvm_started
from gnomepy.java.enums import SchemaType
from gnomepy.java.sbe import get_message
from gnomepy.java.schemas import Schema, get_schema_class, wrap_schema
from gnomepy.java.statics import Scales


_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


class DataStore:
    def __init__(self, data: bytes, schema_type: SchemaType):
        if data[:4] == _ZSTD_MAGIC:
            data = zstandard.ZstdDecompressor().decompress(data)
        self._data: bytes | None = data
        self._schemas: list[Schema] | None = None
        self._schema_type = schema_type

    @classmethod
    def from_bytes(
        cls,
        data: bytes | io.BytesIO | io.RawIOBase | io.BufferedIOBase,
        schema_type: SchemaType,
    ) -> DataStore:
        if isinstance(data, (bytes, bytearray)):
            raw = bytes(data)
        else:
            raw = data.read()
        return cls(raw, schema_type)

    @classmethod
    def from_schemas(cls, schemas: list[Schema], schema_type: SchemaType) -> DataStore:
        instance = object.__new__(cls)
        instance._data = None
        instance._schemas = schemas
        instance._schema_type = schema_type
        return instance

    def __iter__(self) -> Iterator[Schema]:
        if self._schemas is not None:
            yield from self._schemas
            return

        ensure_jvm_started()

        schema_cls = get_schema_class(self._schema_type)
        JavaClass = jpype.JClass(schema_cls._java_class)
        UnsafeBuffer = jpype.JClass("org.agrona.concurrent.UnsafeBuffer")

        proto = JavaClass()
        msg_size = int(proto.totalMessageSize())

        java_bytes = jpype.JArray(jpype.JByte)(self._data)
        src_buf = UnsafeBuffer(java_bytes)

        offset = 0
        data_len = len(self._data)
        while offset + msg_size <= data_len:
            schema = JavaClass()
            schema.buffer.putBytes(0, src_buf, offset, msg_size)
            yield wrap_schema(schema)
            offset += msg_size

    def __len__(self) -> int:
        if self._schemas is not None:
            return len(self._schemas)
        ensure_jvm_started()
        schema_cls = get_schema_class(self._schema_type)
        proto = jpype.JClass(schema_cls._java_class)()
        msg_size = int(proto.totalMessageSize())
        return len(self._data) // msg_size

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
        msg = get_message(self._schema_type)
        print(msg)

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

    def replay(self, callback: Callable[[Schema], None]) -> None:
        for schema in self:
            callback(schema)


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
