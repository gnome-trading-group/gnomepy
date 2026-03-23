from __future__ import annotations

from gnomepy.java.enums import SchemaType, Side


class JavaSchema:
    """Python-friendly wrapper around a live Java Schema object.

    The underlying Java object is accessed via JPype — no serialization occurs.
    Property reads are direct JNI calls to the Java decoder.
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

    def to_dict(self) -> dict:
        """Extract fields into a Python dict. Override in subclasses for typed access."""
        return {
            "schema_type": self.schema_type.value,
            "event_timestamp": self.event_timestamp,
            "sequence_number": self.sequence_number,
        }


class JavaMBOSchema(JavaSchema):
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
            **super().to_dict(),
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


class JavaMBP10Schema(JavaSchema):
    NUM_LEVELS = 10

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
            **super().to_dict(),
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


class JavaMBP1Schema(JavaMBP10Schema):
    """MBP1 has the same decoder layout as MBP10 but with a single level."""
    NUM_LEVELS = 1


class JavaBBOSchema(JavaSchema):
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
            **super().to_dict(),
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


class JavaTradesSchema(JavaSchema):
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
            **super().to_dict(),
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


class JavaOHLCVSchema(JavaSchema):
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
            **super().to_dict(),
            "exchange_id": self.exchange_id,
            "security_id": self.security_id,
            "timestamp_event": self.timestamp_event,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


# Schema type name → wrapper class
_SCHEMA_WRAPPERS: dict[str, type[JavaSchema]] = {
    "MBO": JavaMBOSchema,
    "MBP_10": JavaMBP10Schema,
    "MBP_1": JavaMBP1Schema,
    "BBO_1S": JavaBBOSchema,
    "BBO_1M": JavaBBOSchema,
    "TRADES": JavaTradesSchema,
    "OHLCV_1S": JavaOHLCVSchema,
    "OHLCV_1M": JavaOHLCVSchema,
    "OHLCV_1H": JavaOHLCVSchema,
}


def wrap_schema(java_schema) -> JavaSchema:
    """Wrap a Java Schema object in the appropriate Python wrapper.

    Args:
        java_schema: A Java group.gnometrading.schemas.Schema instance.

    Returns:
        A typed JavaSchema subclass with Python-friendly property accessors.
    """
    schema_type_name = str(java_schema.schemaType.name())
    wrapper_cls = _SCHEMA_WRAPPERS.get(schema_type_name, JavaSchema)
    return wrapper_cls(java_schema)
