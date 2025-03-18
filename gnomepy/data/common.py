from __future__ import annotations

import decimal
import enum
import logging
import warnings
from abc import ABC
from io import BytesIO
from typing import IO, Generator, Protocol, Any, Iterator, Callable

import importlib_resources
import numpy as np
import pandas as pd
import pytz
import zstandard

from gnomepy.data.types import SchemaBase, SchemaType, get_schema_base, DecimalType, FIXED_PRICE_SCALE, FIXED_SIZE_SCALE
from gnomepy.data.sbe import Schema, Message


logger = logging.getLogger(__name__)

def _is_zstandard(reader: IO[bytes]) -> bool:
    """
    Determine if an `IO[bytes]` reader contains zstandard compressed data.

    Parameters
    ----------
    reader : IO[bytes]
        The data to check.

    Returns
    -------
    bool
    """
    reader.seek(0)
    try:
        zstandard.get_frame_parameters(reader.read(18))
    except zstandard.ZstdError:
        return False
    else:
        return True

class Compression(enum.Enum):
    ZSTD = 0
    NONE = 1

class DataSource(ABC):
    """
    Abstract base class for holding schema data.
    """
    @property
    def reader(self) -> IO[bytes]:
        raise NotImplementedError

    @property
    def bytes(self) -> memoryview:
        raise NotImplementedError

class MemoryDataSource(DataSource):
    def __init__(self, source: BytesIO | bytes | IO[bytes]):
        if isinstance(source, bytes):
            initial_data = source
        else:
            source.seek(0)
            initial_data = source.read()

        if len(initial_data) == 0:
            raise ValueError(f"Cannot create data source from empty {type(source).__name__}")
        self.__buffer = BytesIO(initial_data)

    @property
    def reader(self) -> BytesIO:
        self.__buffer.seek(0)
        return self.__buffer

    @property
    def bytes(self) -> memoryview:
        return self.__buffer.getbuffer()

class DataStore:

    def __init__(
            self,
            data_source: DataSource,
            schema_type: SchemaType,
            schema_file_module = "gnomepy.data.sbe",
            schema_file_name = "schema.xml"
    ):
        self._data_source = data_source
        self._schema_type = schema_type
        self._schema_base_type = get_schema_base(self._schema_type)

        buffer = self._data_source.reader
        if _is_zstandard(buffer):
            self._compression = Compression.ZSTD
        else:
            self._compression = Compression.NONE

        with importlib_resources.open_text(schema_file_module, schema_file_name) as f:
            self.schema = Schema.parse(f)
        self._header_size = self.schema.types[self.schema.header_type_name].size()

    def __iter__(self) -> Generator[SchemaBase, None, None]:
        mem = self.bytes
        offset = 0
        body_size = self._schema_metadata.body_size
        while offset < len(mem):
            message = self.schema.decode(mem[offset:])
            parsed = self._schema_base_type.from_message(message)

            yield parsed

            offset += body_size + self._header_size

    def replay(self, callback: Callable[[Any], None]) -> None:
        """
        Replay data by passing records sequentially to the given callback.

        Parameters
        ----------
        callback : callable
            The callback to the data handler.

        """
        for record in self:
            try:
                callback(record)
            except Exception:
                logger.exception("exception while replaying to user callback")
                raise

    def __repr__(self):
        return f"<{self.__class__.__name__}(type={self._schema_type})>"

    @property
    def _schema_metadata(self) -> Message:
        for message in self.schema.messages.values():
            if message.description == self._schema_type.value:
                return message
        raise Exception(f"Invalid schema type: {self._schema_type}")

    @property
    def schema_dtype(self) -> np.dtype:
        metadata = self._schema_metadata
        header_format = f"u{self._header_size}"
        return np.dtype({'names': ['header'] + metadata.field_names, 'formats': [header_format] + metadata.formats})

    @property
    def bytes(self) -> memoryview:
        if self._compression == Compression.ZSTD:
            return memoryview(zstandard.ZstdDecompressor().stream_reader(self._data_source.bytes).readall())
        return self._data_source.bytes

    @property
    def reader(self) -> IO[bytes]:
        if self._compression == Compression.ZSTD:
            return zstandard.ZstdDecompressor().stream_reader(self._data_source.reader)
        return self._data_source.reader

    def to_ndarray(self, count: int | None = None) -> np.ndarray[Any, Any] | NDArrayIterator:
        """
        Return the data as a numpy `ndarray`.

        Parameters
        ----------
        count : int, optional
            If set, instead of returning a single `np.ndarray` a `NDArrayIterator`
            instance will be returned. When iterated, this object will yield
            a `np.ndarray` with at most `count` elements until the entire contents
            of the data store is exhausted. This can be used to process a large
            data store in pieces instead of all at once.

        Returns
        -------
        np.ndarray
        NDArrayIterator
        """
        ndarray_iter = NDArrayIterator(
            reader=self.reader,
            dtype=self.schema_dtype,
            count=count,
        )

        if count is None:
            return next(ndarray_iter, np.empty([0, 1], dtype=self.schema_dtype))

        return ndarray_iter

    def to_df(
            self,
            price_type: DecimalType | str = DecimalType.FLOAT,
            size_type: DecimalType | str = DecimalType.FLOAT,
            pretty_ts: bool = True,
            tz: pytz.BaseTzInfo | str = pytz.UTC,
            replace_nulls: bool = True,
            count: int | None = None,
    ) -> pd.DataFrame | DataFrameIterator:
        """
        Return the data as a `pd.DataFrame`.

        Parameters
        ----------
        price_type : DecimalType or str, default "float"
            The price type to use for price fields.
            If "fixed", prices will have a type of `int` in fixed decimal format; each unit representing 1e-9 or 0.000000001.
            If "float", prices will have a type of `float`.
            If "decimal", prices will be instances of `decimal.Decimal`.
        size_type : DecimalType or str, default "float"
            The size type to use for size fields.
            If "fixed", sizes will have a type of `int` in fixed decimal format; each unit representing 1e-6 or 0.000001.
            If "float", sizes will have a type of `float`.
            If "decimal", sizes will be instances of `decimal.Decimal`.
        pretty_ts : bool, default True
            If all timestamp columns should be converted from UNIX nanosecond
            `int` to tz-aware `pd.Timestamp`. The timezone can be specified using the `tz` parameter.
        tz : pytz.BaseTzInfo or str, default UTC
            If `pretty_ts` is `True`, all timestamps will be converted to the specified timezone.
        replace_nulls : bool, default True
            Replace the null values in the `DataFrame` with `np.nan`.
        count : int, optional
            If set, instead of returning a single `DataFrame` a `DataFrameIterator`
            instance will be returned. When iterated, this object will yield
            a `DataFrame` with at most `count` elements until the entire contents
            of the data store are exhausted. This can be used to process a large
            data store in pieces instead of all at once.

        Returns
        -------
        pd.DataFrame
        DataFrameIterator
        """
        if not isinstance(tz, pytz.BaseTzInfo):
            tz = pytz.timezone(tz)
        if count is None:
            records = iter([self.to_ndarray()])
        else:
            records = self.to_ndarray(count)

        df_iter = DataFrameIterator(
            records=records,
            schema_metadata=self._schema_metadata,
            count=count,
            tz=tz,
            price_type=price_type,
            size_type=size_type,
            replace_nulls=replace_nulls,
            pretty_ts=pretty_ts,
        )
        if count is None:
            return next(df_iter)

        return df_iter

    @classmethod
    def from_bytes(cls, data: BytesIO | bytes | IO[bytes], schema_type: SchemaType) -> DataStore:
        return cls(MemoryDataSource(data), schema_type)


class NDArrayIterator(Protocol):

    def __init__(
        self,
        reader: IO[bytes],
        dtype: np.typing.DTypeLike,
        count: int | None = None,
    ) -> None:
        self._reader = reader
        self._dtype: np.typing.DTypeLike = np.dtype(dtype)
        self._count = count
        self._close_on_next = False

    def __iter__(self) -> NDArrayIterator:
        return self

    def __next__(self) -> np.ndarray[Any, Any]:
        if self._close_on_next:
            raise StopIteration

        if self._count is None:
            read_size = -1
        else:
            read_size = self._dtype.itemsize * max(self._count, 1)

        if buffer := self._reader.read(read_size):
            loose_bytes = len(buffer) % self._dtype.itemsize
            if loose_bytes != 0:
                warnings.warn("Data store file is truncated or contains an incomplete record")
                buffer = buffer[:-loose_bytes]
                self._close_on_next = True

            try:
                return np.frombuffer(
                    buffer=buffer,
                    dtype=self._dtype,
                )
            except ValueError as exc:
                raise Exception("Cannot decode data stream") from exc

        raise StopIteration


class DataFrameIterator:
    def __init__(
        self,
        records: Iterator[np.ndarray[Any, Any]],
        schema_metadata: Message,
        count: int | None,
        tz: pytz.BaseTzInfo,
        price_type: DecimalType = DecimalType.FLOAT,
        size_type: DecimalType = DecimalType.FLOAT,
        replace_nulls: bool = True,
        pretty_ts: bool = True,
    ):
        self._records = records
        self._schema_metadata = schema_metadata
        self._count = count
        self._price_type = price_type
        self._size_type = size_type
        self._replace_nulls = replace_nulls
        self._pretty_ts = pretty_ts
        self._tz = tz

    def __iter__(self) -> DataFrameIterator:
        return self

    def __next__(self) -> pd.DataFrame:
        df = pd.DataFrame(
            next(self._records),
            columns=self._schema_metadata.field_names,
        )
        if self._replace_nulls:
            self._format_nulls(df)

        self._format_decimal(df, 'price', self._price_type, FIXED_PRICE_SCALE)
        self._format_decimal(df, 'size', self._size_type, FIXED_SIZE_SCALE)
        self._format_decimal(df, 'volume', self._size_type, FIXED_SIZE_SCALE)

        if self._pretty_ts:
            self._format_pretty_ts(df)
            self._format_timezone(df)

        return df

    def _format_nulls(self, df: pd.DataFrame):
        for field, na_val in self._schema_metadata.null_fields.items():
            df[field] = df[field].replace(na_val, np.nan)

    def _format_timezone(self, df: pd.DataFrame) -> None:
        for field in self._schema_metadata.fields_by_type('timestamp'):
            df[field] = df[field].dt.tz_convert(self._tz)

    def _format_decimal(
            self,
            df: pd.DataFrame,
            type_name: str,
            decimal_type: DecimalType,
            scale: int,
    ):
        fields = self._schema_metadata.fields_by_type(type_name)

        if decimal_type == DecimalType.DECIMAL:
            df[fields] = (
                df[fields].applymap(decimal.Decimal) / scale
            )
        elif decimal_type == DecimalType.FLOAT:
            df[fields] /= scale
        else:
            return  # do nothing

    def _format_pretty_ts(self, df: pd.DataFrame) -> None:
        for field in self._schema_metadata.fields_by_type('timestamp'):
            df[field] = pd.to_datetime(df[field], utc=True, errors="coerce")
