from __future__ import annotations

import jpype

from gnomepy.java._jvm import ensure_jvm_started
from gnomepy.java.enums import RecordType, Side, JPYPE_KEYWORD_MAP


_LEVEL_FIELDS = [
    ("bid_price", "bidPrice"),
    ("ask_price", "askPrice"),
    ("bid_size", "bidSize"),
    ("ask_size", "askSize"),
    ("bid_count", "bidCount"),
    ("ask_count", "askCount"),
]

_FLAG_SETTERS = {
    "badData": "badData",
    "snapshot": "snapshot",
    "lastMessage": "lastMessage",
    "badTimestampRecv": "badTimestampRecv",
    "marketByPrice": "marketByPrice",
    "topOfBook": "topOfBook",
}


def _set_level_fields(enc, level_kwargs: dict, num_levels: int) -> None:
    for i in range(num_levels):
        for snake_prefix, java_method in _LEVEL_FIELDS:
            val = level_kwargs.get(f"{snake_prefix}_{i}")
            _enc_optional(enc, f"{java_method}{i}", val)


def _set_flags(flags_encoder, flags: list[str]) -> None:
    flags_encoder.clear()
    for flag in flags:
        setter = _FLAG_SETTERS.get(flag)
        if setter:
            getattr(flags_encoder, setter)(True)


def _enc_optional(enc, method: str, value, cast=jpype.JLong) -> None:
    setter = getattr(enc, method)
    if value is not None:
        setter(cast(value))
    else:
        setter(cast(getattr(enc, f"{method}NullValue")()))


def _java_action(name: str):
    Action = jpype.JClass("group.gnometrading.schemas.book.Action")
    return getattr(Action, JPYPE_KEYWORD_MAP.get(name, name))


def _java_side(name: str):
    Side = jpype.JClass("group.gnometrading.schemas.book.Side")
    return getattr(Side, JPYPE_KEYWORD_MAP.get(name, name))


class StreamRecord:
    """Python-friendly wrapper around a live Java StreamRecord: one record of a market-data stream.

    A record names only its stream; which listing (or vendor subject) that is, the session's stream directory says.

    Supports two construction modes:
    - Wrap mode: StreamRecord(java_record) — wraps an existing Java record
    - Subclasses add create mode via keyword arguments
    """

    def __init__(self, java_schema):
        self._java = java_schema

    @property
    def record_type(self) -> RecordType:
        return RecordType.from_java(self._java.recordType)

    @property
    def stream_id(self) -> int:
        return int(self._java.streamId())

    @property
    def is_bad_data(self) -> bool:
        """The record may not reflect its source (a gap, a resync, a lost stream) until a snapshot clears it. While a
        listing's book is bad the OMS refuses new orders on it (``RejectReason.MARKET_DATA_BAD``); resting ones stay."""
        return bool(self._java.isBadData())

    @property
    def is_snapshot(self) -> bool:
        """The record is a whole book, e.g. the one that clears a bad-data spell."""
        return bool(self._java.isSnapshot())

    @property
    def event_timestamp(self) -> int:
        return int(self._java.getEventTimestamp())

    @property
    def sequence_number(self) -> int:
        return int(self._java.getSequenceNumber())

    @property
    def raw(self):
        """Access the underlying Java record directly."""
        return self._java

    def encode(self) -> bytes:
        """Encode the record to SBE bytes."""
        size = int(self._java.totalMessageSize())
        byte_array = jpype.JArray(jpype.JByte)(size)
        self._java.buffer.getBytes(0, byte_array, 0, size)
        return bytes([b & 0xFF for b in byte_array])

class MboSchema(StreamRecord):
    _java_class = "group.gnometrading.schemas.MboSchema"

    def __init__(
        self,
        java_schema=None,
        *,
        stream_id: int | None = None,
        timestamp_event: int | None = None,
        timestamp_sent: int | None = None,
        timestamp_recv: int | None = None,
        order_id: int | None = None,
        price: int | None = None,
        size: int | None = None,
        action: str | None = None,
        side: str | None = None,
        flags: list[str] | None = None,
        sequence: int | None = None,
    ):
        if java_schema is not None:
            self._java = java_schema
            return

        ensure_jvm_started()
        schema = jpype.JClass(self._java_class)()
        enc = schema.encoder
        if stream_id is not None:
            enc.streamId(jpype.JLong(stream_id))
        if timestamp_event is not None:
            enc.timestampEvent(jpype.JLong(timestamp_event))
        if timestamp_sent is not None:
            enc.timestampSent(jpype.JLong(timestamp_sent))
        if timestamp_recv is not None:
            enc.timestampRecv(jpype.JLong(timestamp_recv))
        if order_id is not None:
            enc.orderId(jpype.JLong(order_id))
        _enc_optional(enc, "price", price)
        _enc_optional(enc, "size", size)
        if action is not None:
            enc.action(_java_action(action))
        if side is not None:
            enc.side(_java_side(side))
        if flags is not None:
            _set_flags(enc.flags(), flags)
        if sequence is not None:
            enc.sequence(jpype.JLong(sequence))
        self._java = schema


    @property
    def timestamp_event(self) -> int:
        return int(self._java.decoder.timestampEvent())

    @property
    def timestamp_sent(self) -> int:
        return int(self._java.decoder.timestampSent())

    @property
    def timestamp_recv(self) -> int:
        return int(self._java.decoder.timestampRecv())

    @property
    def order_id(self) -> int:
        return int(self._java.decoder.orderId())

    @property
    def price(self) -> int:
        return int(self._java.decoder.price())

    @property
    def size(self) -> int:
        return int(self._java.decoder.size())

    @property
    def side(self) -> Side:
        return Side.from_java(self._java.decoder.side())

    @property
    def action(self) -> str:
        return str(self._java.decoder.action().name())

    @property
    def sequence(self) -> int:
        return int(self._java.decoder.sequence())

    def to_dict(self) -> dict:
        return {
            "stream_id": self.stream_id,
            "timestamp_event": self.timestamp_event,
            "timestamp_sent": self.timestamp_sent,
            "timestamp_recv": self.timestamp_recv,
            "order_id": self.order_id,
            "price": self.price,
            "size": self.size,
            "side": self.side.value,
            "action": self.action,
            "sequence": self.sequence,
        }


class Mbp10Schema(StreamRecord):
    _java_class = "group.gnometrading.schemas.Mbp10Schema"
    NUM_LEVELS = 10

    def __init__(
        self,
        java_schema=None,
        *,
        stream_id: int | None = None,
        timestamp_event: int | None = None,
        timestamp_sent: int | None = None,
        timestamp_recv: int | None = None,
        price: int | None = None,
        size: int | None = None,
        action: str | None = None,
        side: str | None = None,
        flags: list[str] | None = None,
        sequence: int | None = None,
        depth: int | None = None,
        **level_kwargs,
    ):
        if java_schema is not None:
            self._java = java_schema
            return

        ensure_jvm_started()
        schema = jpype.JClass(self._java_class)()
        enc = schema.encoder
        if stream_id is not None:
            enc.streamId(jpype.JLong(stream_id))
        if timestamp_event is not None:
            enc.timestampEvent(jpype.JLong(timestamp_event))
        if timestamp_sent is not None:
            enc.timestampSent(jpype.JLong(timestamp_sent))
        if timestamp_recv is not None:
            enc.timestampRecv(jpype.JLong(timestamp_recv))
        _enc_optional(enc, "price", price)
        _enc_optional(enc, "size", size)
        if action is not None:
            enc.action(_java_action(action))
        if side is not None:
            enc.side(_java_side(side))
        if flags is not None:
            _set_flags(enc.flags(), flags)
        if sequence is not None:
            enc.sequence(jpype.JLong(sequence))
        _enc_optional(enc, "depth", depth, cast=jpype.JShort)
        _set_level_fields(enc, level_kwargs, self.NUM_LEVELS)
        self._java = schema


    @property
    def timestamp_event(self) -> int:
        return int(self._java.decoder.timestampEvent())

    @property
    def timestamp_sent(self) -> int:
        return int(self._java.decoder.timestampSent())

    @property
    def timestamp_recv(self) -> int:
        return int(self._java.decoder.timestampRecv())

    @property
    def price(self) -> int:
        return int(self._java.decoder.price())

    @property
    def size(self) -> int:
        return int(self._java.decoder.size())

    @property
    def side(self) -> Side:
        return Side.from_java(self._java.decoder.side())

    @property
    def action(self) -> str:
        return str(self._java.decoder.action().name())

    @property
    def depth(self) -> int:
        return int(self._java.decoder.depth())

    @property
    def sequence(self) -> int:
        return int(self._java.decoder.sequence())

    def bid_price(self, level: int) -> int:
        return int(getattr(self._java.decoder, f"bidPrice{level}")())

    def ask_price(self, level: int) -> int:
        return int(getattr(self._java.decoder, f"askPrice{level}")())

    def bid_size(self, level: int) -> int:
        return int(getattr(self._java.decoder, f"bidSize{level}")())

    def ask_size(self, level: int) -> int:
        return int(getattr(self._java.decoder, f"askSize{level}")())

    def bid_count(self, level: int) -> int:
        return int(getattr(self._java.decoder, f"bidCount{level}")())

    def ask_count(self, level: int) -> int:
        return int(getattr(self._java.decoder, f"askCount{level}")())

    def to_dict(self) -> dict:
        d = {
            "stream_id": self.stream_id,
            "timestamp_event": self.timestamp_event,
            "timestamp_sent": self.timestamp_sent,
            "timestamp_recv": self.timestamp_recv,
            "price": self.price,
            "size": self.size,
            "side": self.side.value,
            "action": self.action,
            "depth": self.depth,
            "sequence": self.sequence,
        }
        for i in range(self.NUM_LEVELS):
            d[f"bid_price_{i}"] = self.bid_price(i)
            d[f"ask_price_{i}"] = self.ask_price(i)
            d[f"bid_size_{i}"] = self.bid_size(i)
            d[f"ask_size_{i}"] = self.ask_size(i)
            d[f"bid_count_{i}"] = self.bid_count(i)
            d[f"ask_count_{i}"] = self.ask_count(i)
        return d


class Mbp1Schema(Mbp10Schema):
    """MBP1 has the same decoder layout as MBP10 but with a single level."""
    _java_class = "group.gnometrading.schemas.Mbp1Schema"
    NUM_LEVELS = 1

    def __init__(
        self,
        java_schema=None,
        *,
        stream_id: int | None = None,
        timestamp_event: int | None = None,
        timestamp_sent: int | None = None,
        timestamp_recv: int | None = None,
        price: int | None = None,
        size: int | None = None,
        action: str | None = None,
        side: str | None = None,
        flags: list[str] | None = None,
        sequence: int | None = None,
        depth: int | None = None,
        **level_kwargs,
    ):
        if java_schema is not None:
            self._java = java_schema
            return

        ensure_jvm_started()
        schema = jpype.JClass(self._java_class)()
        enc = schema.encoder
        if stream_id is not None:
            enc.streamId(jpype.JLong(stream_id))
        if timestamp_event is not None:
            enc.timestampEvent(jpype.JLong(timestamp_event))
        if timestamp_sent is not None:
            enc.timestampSent(jpype.JLong(timestamp_sent))
        if timestamp_recv is not None:
            enc.timestampRecv(jpype.JLong(timestamp_recv))
        _enc_optional(enc, "price", price)
        _enc_optional(enc, "size", size)
        if action is not None:
            enc.action(_java_action(action))
        if side is not None:
            enc.side(_java_side(side))
        if flags is not None:
            _set_flags(enc.flags(), flags)
        if sequence is not None:
            enc.sequence(jpype.JLong(sequence))
        _enc_optional(enc, "depth", depth, cast=jpype.JShort)
        _set_level_fields(enc, level_kwargs, 1)
        self._java = schema


class BboSchema(StreamRecord):
    def __init__(
        self,
        java_schema=None,
        *,
        stream_id: int | None = None,
        timestamp_event: int | None = None,
        timestamp_recv: int | None = None,
        price: int | None = None,
        size: int | None = None,
        side: str | None = None,
        sequence: int | None = None,
        bid_price_0: int | None = None,
        ask_price_0: int | None = None,
        bid_size_0: int | None = None,
        ask_size_0: int | None = None,
        bid_count_0: int | None = None,
        ask_count_0: int | None = None,
    ):
        if java_schema is not None:
            self._java = java_schema
            return

        ensure_jvm_started()
        schema = jpype.JClass(type(self)._java_class)()
        enc = schema.encoder
        if stream_id is not None:
            enc.streamId(jpype.JLong(stream_id))
        if timestamp_event is not None:
            enc.timestampEvent(jpype.JLong(timestamp_event))
        if timestamp_recv is not None:
            enc.timestampRecv(jpype.JLong(timestamp_recv))
        _enc_optional(enc, "price", price)
        _enc_optional(enc, "size", size)
        if side is not None:
            enc.side(_java_side(side))
        if sequence is not None:
            enc.sequence(jpype.JLong(sequence))
        _enc_optional(enc, "bidPrice0", bid_price_0)
        _enc_optional(enc, "askPrice0", ask_price_0)
        _enc_optional(enc, "bidSize0", bid_size_0)
        _enc_optional(enc, "askSize0", ask_size_0)
        _enc_optional(enc, "bidCount0", bid_count_0)
        _enc_optional(enc, "askCount0", ask_count_0)
        self._java = schema


    @property
    def timestamp_event(self) -> int:
        return int(self._java.decoder.timestampEvent())

    @property
    def timestamp_recv(self) -> int:
        return int(self._java.decoder.timestampRecv())

    @property
    def price(self) -> int:
        return int(self._java.decoder.price())

    @property
    def size(self) -> int:
        return int(self._java.decoder.size())

    @property
    def side(self) -> Side:
        return Side.from_java(self._java.decoder.side())

    @property
    def sequence(self) -> int:
        return int(self._java.decoder.sequence())

    @property
    def bid_price(self) -> int:
        return int(self._java.decoder.bidPrice0())

    @property
    def ask_price(self) -> int:
        return int(self._java.decoder.askPrice0())

    @property
    def bid_size(self) -> int:
        return int(self._java.decoder.bidSize0())

    @property
    def ask_size(self) -> int:
        return int(self._java.decoder.askSize0())

    @property
    def bid_count(self) -> int:
        return int(self._java.decoder.bidCount0())

    @property
    def ask_count(self) -> int:
        return int(self._java.decoder.askCount0())

    def to_dict(self) -> dict:
        return {
            "stream_id": self.stream_id,
            "timestamp_event": self.timestamp_event,
            "timestamp_recv": self.timestamp_recv,
            "price": self.price,
            "size": self.size,
            "side": self.side.value,
            "sequence": self.sequence,
            "bid_price": self.bid_price,
            "ask_price": self.ask_price,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "bid_count": self.bid_count,
            "ask_count": self.ask_count,
        }


class Bbo1sSchema(BboSchema):
    _java_class = "group.gnometrading.schemas.Bbo1sSchema"


class Bbo1mSchema(BboSchema):
    _java_class = "group.gnometrading.schemas.Bbo1mSchema"


class TradesSchema(StreamRecord):
    _java_class = "group.gnometrading.schemas.TradesSchema"

    def __init__(
        self,
        java_schema=None,
        *,
        stream_id: int | None = None,
        timestamp_event: int | None = None,
        timestamp_sent: int | None = None,
        timestamp_recv: int | None = None,
        price: int | None = None,
        size: int | None = None,
        action: str | None = None,
        side: str | None = None,
        sequence: int | None = None,
        depth: int | None = None,
    ):
        if java_schema is not None:
            self._java = java_schema
            return

        ensure_jvm_started()
        schema = jpype.JClass(self._java_class)()
        enc = schema.encoder
        if stream_id is not None:
            enc.streamId(jpype.JLong(stream_id))
        if timestamp_event is not None:
            enc.timestampEvent(jpype.JLong(timestamp_event))
        if timestamp_sent is not None:
            enc.timestampSent(jpype.JLong(timestamp_sent))
        if timestamp_recv is not None:
            enc.timestampRecv(jpype.JLong(timestamp_recv))
        _enc_optional(enc, "price", price)
        _enc_optional(enc, "size", size)
        if action is not None:
            enc.action(_java_action(action))
        if side is not None:
            enc.side(_java_side(side))
        if sequence is not None:
            enc.sequence(jpype.JLong(sequence))
        _enc_optional(enc, "depth", depth, cast=jpype.JShort)
        self._java = schema


    @property
    def timestamp_event(self) -> int:
        return int(self._java.decoder.timestampEvent())

    @property
    def timestamp_sent(self) -> int:
        return int(self._java.decoder.timestampSent())

    @property
    def timestamp_recv(self) -> int:
        return int(self._java.decoder.timestampRecv())

    @property
    def price(self) -> int:
        return int(self._java.decoder.price())

    @property
    def size(self) -> int:
        return int(self._java.decoder.size())

    @property
    def side(self) -> Side:
        return Side.from_java(self._java.decoder.side())

    @property
    def action(self) -> str:
        return str(self._java.decoder.action().name())

    @property
    def sequence(self) -> int:
        return int(self._java.decoder.sequence())

    @property
    def depth(self) -> int:
        return int(self._java.decoder.depth())

    def to_dict(self) -> dict:
        return {
            "stream_id": self.stream_id,
            "timestamp_event": self.timestamp_event,
            "timestamp_sent": self.timestamp_sent,
            "timestamp_recv": self.timestamp_recv,
            "price": self.price,
            "size": self.size,
            "side": self.side.value,
            "action": self.action,
            "sequence": self.sequence,
            "depth": self.depth,
        }


class OhlcvSchema(StreamRecord):
    def __init__(
        self,
        java_schema=None,
        *,
        stream_id: int | None = None,
        timestamp_event: int | None = None,
        open: int | None = None,
        high: int | None = None,
        low: int | None = None,
        close: int | None = None,
        volume: int | None = None,
    ):
        if java_schema is not None:
            self._java = java_schema
            return

        ensure_jvm_started()
        schema = jpype.JClass(type(self)._java_class)()
        enc = schema.encoder
        if stream_id is not None:
            enc.streamId(jpype.JLong(stream_id))
        if timestamp_event is not None:
            enc.timestampEvent(jpype.JLong(timestamp_event))
        _enc_optional(enc, "open", open)
        _enc_optional(enc, "high", high)
        _enc_optional(enc, "low", low)
        _enc_optional(enc, "close", close)
        if volume is not None:
            enc.volume(jpype.JLong(volume))
        self._java = schema


    @property
    def timestamp_event(self) -> int:
        return int(self._java.decoder.timestampEvent())

    @property
    def open(self) -> int:
        return int(self._java.decoder.open())

    @property
    def high(self) -> int:
        return int(self._java.decoder.high())

    @property
    def low(self) -> int:
        return int(self._java.decoder.low())

    @property
    def close(self) -> int:
        return int(self._java.decoder.close())

    @property
    def volume(self) -> int:
        return int(self._java.decoder.volume())

    def to_dict(self) -> dict:
        return {
            "stream_id": self.stream_id,
            "timestamp_event": self.timestamp_event,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class Ohlcv1sSchema(OhlcvSchema):
    _java_class = "group.gnometrading.schemas.Ohlcv1sSchema"


class Ohlcv1mSchema(OhlcvSchema):
    _java_class = "group.gnometrading.schemas.Ohlcv1mSchema"


class Ohlcv1hSchema(OhlcvSchema):
    _java_class = "group.gnometrading.schemas.Ohlcv1hSchema"


# Record type name → wrapper class
_RECORD_WRAPPERS: dict[str, type[StreamRecord]] = {
    "MBO": MboSchema,
    "MBP_10": Mbp10Schema,
    "MBP_1": Mbp1Schema,
    "BBO_1S": Bbo1sSchema,
    "BBO_1M": Bbo1mSchema,
    "TRADES": TradesSchema,
    "OHLCV_1S": Ohlcv1sSchema,
    "OHLCV_1M": Ohlcv1mSchema,
    "OHLCV_1H": Ohlcv1hSchema,
}


def get_record_class(record_type: RecordType) -> type[StreamRecord]:
    """Return the Python wrapper class for a given RecordType."""
    return _RECORD_WRAPPERS[record_type.name]


def wrap_record(java_record) -> StreamRecord:
    """Wrap a Java StreamRecord in the appropriate Python wrapper.

    Args:
        java_record: A Java group.gnometrading.schemas.StreamRecord instance.

    Returns:
        A typed StreamRecord subclass with Python-friendly property accessors.
    """
    record_type_name = str(java_record.recordType.name())
    wrapper_cls = _RECORD_WRAPPERS.get(record_type_name, StreamRecord)
    return wrapper_cls(java_record)
