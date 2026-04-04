from __future__ import annotations

import jpype

from gnomepy.java._jvm import ensure_jvm_started
from gnomepy.java.enums import SchemaType, Side, JPYPE_KEYWORD_MAP


_LEVEL_FIELDS = [
    ("bid_price", "bidPrice"),
    ("ask_price", "askPrice"),
    ("bid_size", "bidSize"),
    ("ask_size", "askSize"),
    ("bid_count", "bidCount"),
    ("ask_count", "askCount"),
]

_FLAG_SETTERS = {
    "lastMessage": "lastMessage",
    "topOfBook": "topOfBook",
    "snapshot": "snapshot",
    "marketByPrice": "marketByPrice",
    "badTimestampRecv": "badTimestampRecv",
    "maybeBadBook": "maybeBadBook",
}


def _set_level_fields(enc, level_kwargs: dict, num_levels: int) -> None:
    for i in range(num_levels):
        for snake_prefix, java_method in _LEVEL_FIELDS:
            val = level_kwargs.get(f"{snake_prefix}_{i}")
            if val is not None:
                getattr(enc, f"{java_method}{i}")(jpype.JLong(val))


def _set_flags(flags_encoder, flags: list[str]) -> None:
    flags_encoder.clear()
    for flag in flags:
        setter = _FLAG_SETTERS.get(flag)
        if setter:
            getattr(flags_encoder, setter)(True)


def _java_action(name: str):
    Action = jpype.JClass("group.gnometrading.schemas.Action")
    return getattr(Action, JPYPE_KEYWORD_MAP.get(name, name))


def _java_side(name: str):
    Side = jpype.JClass("group.gnometrading.schemas.Side")
    return getattr(Side, JPYPE_KEYWORD_MAP.get(name, name))


class Schema:
    """Python-friendly wrapper around a live Java Schema object.

    Supports two construction modes:
    - Wrap mode: Schema(java_schema) — wraps an existing Java schema object
    - Subclasses add create mode via keyword arguments
    """

    def __init__(self, java_schema):
        self._java = java_schema

    @property
    def schema_type(self) -> SchemaType:
        return SchemaType.from_java(self._java.schemaType)

    @property
    def event_timestamp(self) -> int:
        return int(self._java.getEventTimestamp())

    @property
    def sequence_number(self) -> int:
        return int(self._java.getSequenceNumber())

    @property
    def raw(self):
        """Access the underlying Java Schema object directly."""
        return self._java

    def encode(self) -> bytes:
        """Encode the schema to SBE bytes."""
        size = int(self._java.totalMessageSize())
        byte_array = jpype.JArray(jpype.JByte)(size)
        self._java.buffer.getBytes(0, byte_array, 0, size)
        return bytes([b & 0xFF for b in byte_array])

class MboSchema(Schema):
    _java_class = "group.gnometrading.schemas.MBOSchema"

    def __init__(
        self,
        java_schema=None,
        *,
        exchange_id: int | None = None,
        security_id: int | None = None,
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
        if exchange_id is not None:
            enc.exchangeId(int(exchange_id))
        if security_id is not None:
            enc.securityId(jpype.JLong(security_id))
        if timestamp_event is not None:
            enc.timestampEvent(jpype.JLong(timestamp_event))
        if timestamp_sent is not None:
            enc.timestampSent(jpype.JLong(timestamp_sent))
        if timestamp_recv is not None:
            enc.timestampRecv(jpype.JLong(timestamp_recv))
        if order_id is not None:
            enc.orderId(jpype.JLong(order_id))
        if price is not None:
            enc.price(jpype.JLong(price))
        if size is not None:
            enc.size(jpype.JLong(size))
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
    def exchange_id(self) -> int:
        return int(self._java.decoder.exchangeId())

    @property
    def security_id(self) -> int:
        return int(self._java.decoder.securityId())

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
            "exchange_id": self.exchange_id,
            "security_id": self.security_id,
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


class Mbp10Schema(Schema):
    _java_class = "group.gnometrading.schemas.MBP10Schema"
    NUM_LEVELS = 10

    def __init__(
        self,
        java_schema=None,
        *,
        exchange_id: int | None = None,
        security_id: int | None = None,
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
        if exchange_id is not None:
            enc.exchangeId(int(exchange_id))
        if security_id is not None:
            enc.securityId(jpype.JLong(security_id))
        if timestamp_event is not None:
            enc.timestampEvent(jpype.JLong(timestamp_event))
        if timestamp_sent is not None:
            enc.timestampSent(jpype.JLong(timestamp_sent))
        if timestamp_recv is not None:
            enc.timestampRecv(jpype.JLong(timestamp_recv))
        if price is not None:
            enc.price(jpype.JLong(price))
        if size is not None:
            enc.size(jpype.JLong(size))
        if action is not None:
            enc.action(_java_action(action))
        if side is not None:
            enc.side(_java_side(side))
        if flags is not None:
            _set_flags(enc.flags(), flags)
        if sequence is not None:
            enc.sequence(jpype.JLong(sequence))
        if depth is not None:
            enc.depth(jpype.JShort(depth))
        _set_level_fields(enc, level_kwargs, self.NUM_LEVELS)
        self._java = schema

    @property
    def exchange_id(self) -> int:
        return int(self._java.decoder.exchangeId())

    @property
    def security_id(self) -> int:
        return int(self._java.decoder.securityId())

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
            "exchange_id": self.exchange_id,
            "security_id": self.security_id,
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
    _java_class = "group.gnometrading.schemas.MBP1Schema"
    NUM_LEVELS = 1

    def __init__(
        self,
        java_schema=None,
        *,
        exchange_id: int | None = None,
        security_id: int | None = None,
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
        if exchange_id is not None:
            enc.exchangeId(int(exchange_id))
        if security_id is not None:
            enc.securityId(jpype.JLong(security_id))
        if timestamp_event is not None:
            enc.timestampEvent(jpype.JLong(timestamp_event))
        if timestamp_sent is not None:
            enc.timestampSent(jpype.JLong(timestamp_sent))
        if timestamp_recv is not None:
            enc.timestampRecv(jpype.JLong(timestamp_recv))
        if price is not None:
            enc.price(jpype.JLong(price))
        if size is not None:
            enc.size(jpype.JLong(size))
        if action is not None:
            enc.action(_java_action(action))
        if side is not None:
            enc.side(_java_side(side))
        if flags is not None:
            _set_flags(enc.flags(), flags)
        if sequence is not None:
            enc.sequence(jpype.JLong(sequence))
        if depth is not None:
            enc.depth(jpype.JShort(depth))
        _set_level_fields(enc, level_kwargs, 1)
        self._java = schema


class BboSchema(Schema):
    def __init__(
        self,
        java_schema=None,
        *,
        exchange_id: int | None = None,
        security_id: int | None = None,
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
        if exchange_id is not None:
            enc.exchangeId(int(exchange_id))
        if security_id is not None:
            enc.securityId(jpype.JLong(security_id))
        if timestamp_event is not None:
            enc.timestampEvent(jpype.JLong(timestamp_event))
        if timestamp_recv is not None:
            enc.timestampRecv(jpype.JLong(timestamp_recv))
        if price is not None:
            enc.price(jpype.JLong(price))
        if size is not None:
            enc.size(jpype.JLong(size))
        if side is not None:
            enc.side(_java_side(side))
        if sequence is not None:
            enc.sequence(jpype.JLong(sequence))
        if bid_price_0 is not None:
            enc.bidPrice0(jpype.JLong(bid_price_0))
        if ask_price_0 is not None:
            enc.askPrice0(jpype.JLong(ask_price_0))
        if bid_size_0 is not None:
            enc.bidSize0(jpype.JLong(bid_size_0))
        if ask_size_0 is not None:
            enc.askSize0(jpype.JLong(ask_size_0))
        if bid_count_0 is not None:
            enc.bidCount0(jpype.JLong(bid_count_0))
        if ask_count_0 is not None:
            enc.askCount0(jpype.JLong(ask_count_0))
        self._java = schema

    @property
    def exchange_id(self) -> int:
        return int(self._java.decoder.exchangeId())

    @property
    def security_id(self) -> int:
        return int(self._java.decoder.securityId())

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
            "exchange_id": self.exchange_id,
            "security_id": self.security_id,
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
    _java_class = "group.gnometrading.schemas.BBO1SSchema"


class Bbo1mSchema(BboSchema):
    _java_class = "group.gnometrading.schemas.BBO1MSchema"


class TradesSchema(Schema):
    _java_class = "group.gnometrading.schemas.TradesSchema"

    def __init__(
        self,
        java_schema=None,
        *,
        exchange_id: int | None = None,
        security_id: int | None = None,
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
        if exchange_id is not None:
            enc.exchangeId(int(exchange_id))
        if security_id is not None:
            enc.securityId(jpype.JLong(security_id))
        if timestamp_event is not None:
            enc.timestampEvent(jpype.JLong(timestamp_event))
        if timestamp_sent is not None:
            enc.timestampSent(jpype.JLong(timestamp_sent))
        if timestamp_recv is not None:
            enc.timestampRecv(jpype.JLong(timestamp_recv))
        if price is not None:
            enc.price(jpype.JLong(price))
        if size is not None:
            enc.size(jpype.JLong(size))
        if action is not None:
            enc.action(_java_action(action))
        if side is not None:
            enc.side(_java_side(side))
        if sequence is not None:
            enc.sequence(jpype.JLong(sequence))
        if depth is not None:
            enc.depth(jpype.JShort(depth))
        self._java = schema

    @property
    def exchange_id(self) -> int:
        return int(self._java.decoder.exchangeId())

    @property
    def security_id(self) -> int:
        return int(self._java.decoder.securityId())

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
            "exchange_id": self.exchange_id,
            "security_id": self.security_id,
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


class OhlcvSchema(Schema):
    def __init__(
        self,
        java_schema=None,
        *,
        exchange_id: int | None = None,
        security_id: int | None = None,
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
        if exchange_id is not None:
            enc.exchangeId(int(exchange_id))
        if security_id is not None:
            enc.securityId(jpype.JLong(security_id))
        if timestamp_event is not None:
            enc.timestampEvent(jpype.JLong(timestamp_event))
        if open is not None:
            enc.open(jpype.JLong(open))
        if high is not None:
            enc.high(jpype.JLong(high))
        if low is not None:
            enc.low(jpype.JLong(low))
        if close is not None:
            enc.close(jpype.JLong(close))
        if volume is not None:
            enc.volume(jpype.JLong(volume))
        self._java = schema

    @property
    def exchange_id(self) -> int:
        return int(self._java.decoder.exchangeId())

    @property
    def security_id(self) -> int:
        return int(self._java.decoder.securityId())

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
            "exchange_id": self.exchange_id,
            "security_id": self.security_id,
            "timestamp_event": self.timestamp_event,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class Ohlcv1sSchema(OhlcvSchema):
    _java_class = "group.gnometrading.schemas.OHLCV1SSchema"


class Ohlcv1mSchema(OhlcvSchema):
    _java_class = "group.gnometrading.schemas.OHLCV1MSchema"


class Ohlcv1hSchema(OhlcvSchema):
    _java_class = "group.gnometrading.schemas.OHLCV1HSchema"


# Schema type name → wrapper class
_SCHEMA_WRAPPERS: dict[str, type[Schema]] = {
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


def get_schema_class(schema_type: SchemaType) -> type[Schema]:
    """Return the Python Schema subclass for a given SchemaType."""
    return _SCHEMA_WRAPPERS[schema_type.name]


def wrap_schema(java_schema) -> Schema:
    """Wrap a Java Schema object in the appropriate Python wrapper.

    Args:
        java_schema: A Java group.gnometrading.schemas.Schema instance.

    Returns:
        A typed Schema subclass with Python-friendly property accessors.
    """
    schema_type_name = str(java_schema.schemaType.name())
    wrapper_cls = _SCHEMA_WRAPPERS.get(schema_type_name, Schema)
    return wrapper_cls(java_schema)
