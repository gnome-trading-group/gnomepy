from __future__ import annotations

from enum import Enum


# Maps Java enum constant names to their JPype attribute names when they conflict with Python keywords.
JPYPE_KEYWORD_MAP = {"None": "None_"}


def _java_enum(java_class_name: str, java_value_name: str):
    """Lazily resolve a Java enum value via JPype."""
    import jpype
    cls = jpype.JClass(java_class_name)
    return getattr(cls, JPYPE_KEYWORD_MAP.get(java_value_name, java_value_name))

class Action(Enum):
    ADD = "Add"
    CANCEL = "Cancel"
    MODIFY = "Modify"
    CLEAR = "Clear"
    TRADE = "Trade"
    FILL = "Fill"
    NONE = "None_"

    def to_java(self):
        return _java_enum("group.gnometrading.schemas.Action", self.value)

    @classmethod
    def from_java(cls, java_enum) -> Action:
        name = str(java_enum.name())
        return cls(JPYPE_KEYWORD_MAP.get(name, name))


class Side(Enum):
    BID = "Bid"
    ASK = "Ask"
    NONE = "None_"

    def to_java(self):
        return _java_enum("group.gnometrading.schemas.Side", self.value)

    @classmethod
    def from_java(cls, java_enum) -> Side:
        name = str(java_enum.name())
        return cls(JPYPE_KEYWORD_MAP.get(name, name))


class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"

    def to_java(self):
        return _java_enum("group.gnometrading.schemas.OrderType", self.value)

    @classmethod
    def from_java(cls, java_enum) -> OrderType:
        return cls(str(java_enum.name()))


class TimeInForce(Enum):
    GOOD_TILL_CANCELED = "GOOD_TILL_CANCELED"
    IMMEDIATE_OR_CANCELED = "IMMEDIATE_OR_CANCELED"
    FILL_OR_KILL = "FILL_OR_KILL"

    def to_java(self):
        return _java_enum("group.gnometrading.schemas.TimeInForce", self.value)

    @classmethod
    def from_java(cls, java_enum) -> TimeInForce:
        return cls(str(java_enum.name()))


class ExecType(Enum):
    NEW = "NEW"
    CANCEL = "CANCEL"
    FILL = "FILL"
    PARTIAL_FILL = "PARTIAL_FILL"
    REJECT = "REJECT"
    CANCEL_REJECT = "CANCEL_REJECT"
    EXPIRE = "EXPIRE"

    def to_java(self):
        return _java_enum("group.gnometrading.schemas.ExecType", self.value)

    @classmethod
    def from_java(cls, java_enum) -> ExecType:
        return cls(str(java_enum.name()))


class OrderStatus(Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

    def to_java(self):
        return _java_enum("group.gnometrading.schemas.OrderStatus", self.value)

    @classmethod
    def from_java(cls, java_enum) -> OrderStatus:
        return cls(str(java_enum.name()))


class SchemaType(Enum):
    MBO = "mbo"
    MBP_10 = "mbp-10"
    MBP_1 = "mbp-1"
    BBO_1S = "bbo-1s"
    BBO_1M = "bbo-1m"
    TRADES = "trades"
    OHLCV_1S = "ohlcv-1s"
    OHLCV_1M = "ohlcv-1m"
    OHLCV_1H = "ohlcv-1h"

    def to_java(self):
        """Convert to Java SchemaType enum."""
        import jpype
        JavaSchemaType = jpype.JClass("group.gnometrading.schemas.SchemaType")
        return JavaSchemaType.findById(self.value)

    @classmethod
    def from_java(cls, java_enum) -> SchemaType:
        return cls(str(java_enum.getIdentifier()))
