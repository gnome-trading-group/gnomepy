from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from enum import IntFlag, StrEnum
from typing import Type, Self
import importlib_resources
import re
from gnomepy.data.sbe import DecodedMessage, Schema

FIXED_PRICE_SCALE = 1e9
FIXED_SIZE_SCALE = 1e6

class OrderType(StrEnum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"

class TimeInForce(StrEnum):
    GTC = "GOOD_TILL_CANCELLED"
    IOC = "IMMEDIATE_OR_CANCELED"
    FOK = "FILL_OR_KILL"

class ExecType(StrEnum):
    NEW = "NEW"
    CANCELED = "CANCELED"
    TRADE = "TRADE"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class OrderStatus(StrEnum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

@dataclass
class Order:
    exchange_id: int
    security_id: int
    client_oid: str | None
    price: int
    size: int
    side: str
    order_type: OrderType
    time_in_force: TimeInForce

@dataclass
class OrderExecutionReport:
    exchange_id: int
    security_id: int
    client_oid: str
    exec_type: ExecType
    order_status: OrderStatus
    filled_qty: int
    filled_price: int
    cumulative_qty: int
    leaves_qty: int
    timestamp_event: int
    timestamp_recv: int
    fee: float

@dataclass
class CancelOrder:
    exchange_id: int
    security_id: int
    client_oid: str

class SchemaType(StrEnum):
    MBO = "mbo"
    MBP_10 = "mbp-10"
    MBP_1 = "mbp-1"
    BBO_1S = "bbo-1s"
    BBO_1M = "bbo-1m"
    TRADES = "trades"
    OHLCV_1S = "ohlcv-1s"
    OHLCV_1M = "ohlcv-1m"
    OHLCV_1H = "ohlcv-1h"

class DecimalType(StrEnum):
    FIXED = "fixed"
    FLOAT = "float"
    DECIMAL = "decimal"

class MarketUpdateFlags(IntFlag):
    """
    Represents record flags.

    F_LAST
        Marks the last record in a single event for a given `security_id`.
    F_TOB
        Indicates a top-of-book message, not an individual order.
    F_SNAPSHOT
        Message sourced from a replay, such as a snapshot server.
    F_MBP
        Aggregated price level message, not an individual order.
    F_BAD_TS_RECV
        The `ts_recv` value is inaccurate (clock issues or reordering).
    F_MAYBE_BAD_BOOK
        Indicates an unrecoverable gap was detected in the channel.

    Other bits are reserved and have no current meaning.

    """

    F_LAST = 128
    F_TOB = 64
    F_SNAPSHOT = 32
    F_MBP = 16
    F_BAD_TS_RECV = 8
    F_MAYBE_BAD_BOOK = 4

@dataclass
class BidAskPair:
    bid_px: int
    ask_px: int
    bid_sz: int
    ask_sz: int
    bid_ct: int
    ask_ct: int

    @classmethod
    def from_dict(cls, body: dict, idx: int):
        return cls(
            body[f"bidPrice{idx}"], body[f"askPrice{idx}"],
            body[f"bidSize{idx}"], body[f"askSize{idx}"],
            body[f"bidCount{idx}"], body[f"askCount{idx}"]
        )

    @classmethod
    def to_dict(cls, item: Self | None, idx: int) -> dict:
        if item is None:
            return {
                f"bidPrice{idx}": None,
                f"askPrice{idx}": None,
                f"bidSize{idx}": None,
                f"askSize{idx}": None,
                f"bidCount{idx}": None,
                f"askCount{idx}": None,
            }
        return {
            f"bidPrice{idx}": item.bid_px,
            f"askPrice{idx}": item.ask_px,
            f"bidSize{idx}": item.bid_sz,
            f"askSize{idx}": item.ask_sz,
            f"bidCount{idx}": item.bid_ct,
            f"askCount{idx}": item.ask_ct,
        }

    @property
    def pretty_bid_px(self) -> float:
        return self.bid_px / FIXED_PRICE_SCALE

    @property
    def pretty_ask_px(self) -> float:
        return self.ask_px / FIXED_PRICE_SCALE

class SchemaBase(ABC):
    _schema: Schema | None = None

    @classmethod
    def _load_schema(cls) -> Schema:
        """Load and cache the SBE schema"""
        if cls._schema is None:
            with importlib_resources.open_text("gnomepy.data.sbe", "schema.xml") as f:
                cls._schema = Schema.parse(f)
        return cls._schema

    @classmethod
    @abstractmethod
    def _get_message_id(cls) -> int:
        """Return the SBE message ID for this schema type"""
        raise NotImplementedError

    @staticmethod
    def _to_camel_case(snake_str: str) -> str:
        """Convert snake_case to camelCase."""
        components = snake_str.split('_')
        return components[0] + ''.join(x.title() for x in components[1:])

    def _to_sbe_dict(self) -> dict:
        """Convert dataclass to SBE field dictionary.

        Default implementation: automatically convert all fields to camelCase.
        Subclasses can override _customize_sbe_dict() for special handling.
        """
        result = {}
        for field in fields(self):
            camel_name = self._to_camel_case(field.name)
            value = getattr(self, field.name)
            result[camel_name] = value

        return self._customize_sbe_dict(result)

    def _customize_sbe_dict(self, sbe_dict: dict) -> dict:
        """Override in subclasses to customize the SBE dict (e.g., flatten levels).

        Default implementation: return dict as-is.
        """
        return sbe_dict

    @staticmethod
    def _from_camel_case(camel_str: str) -> str:
        """Convert camelCase to snake_case."""
        snake = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_str)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', snake).lower()

    @classmethod
    def from_message(cls, message: DecodedMessage):
        """Create instance from decoded SBE message.

        Default implementation: automatically map camelCase fields to snake_case.
        Subclasses can override _customize_from_message() for special handling.
        """
        body = message.value

        kwargs = {}
        for field in fields(cls):
            camel_name = cls._to_camel_case(field.name)
            if camel_name in body:
                kwargs[field.name] = body[camel_name]

        return cls._customize_from_message(body, kwargs)

    @classmethod
    def _customize_from_message(cls, body: dict, kwargs: dict):
        """Override in subclasses to customize from_message (e.g., reconstruct levels).

        Default implementation: create instance with kwargs.
        """
        return cls(**kwargs)

    def encode(self) -> bytes:
        """Encode the message to bytes with SBE header"""
        schema = self._load_schema()
        message = schema.messages[self._get_message_id()]
        sbe_dict = self._to_sbe_dict()
        return schema.encode(message, sbe_dict)

class SizeMixin:
    @property
    def pretty_size(self):
        if hasattr(self, 'size'):
            return self.size / FIXED_SIZE_SCALE

class PriceMixin:
    @property
    def pretty_price(self):
        if hasattr(self, 'price'):
            return self.price / FIXED_PRICE_SCALE

@dataclass
class MBO(SchemaBase, PriceMixin, SizeMixin):
    exchange_id: int
    security_id: int
    timestamp_event: int
    timestamp_sent: int | None
    timestamp_recv: int
    order_id: int
    price: int
    size: int
    action: str
    side: str
    flags: list[str]
    sequence: int | None

    @classmethod
    def _get_message_id(cls) -> int:
        return 1


@dataclass
class MBP10(SchemaBase, PriceMixin, SizeMixin):
    exchange_id: int
    security_id: int
    timestamp_event: int
    timestamp_sent: int | None
    timestamp_recv: int
    price: int | None
    size: int | None
    action: str
    side: str | None
    flags: list[str]
    sequence: int | None
    depth: int | None
    levels: list[BidAskPair]

    @classmethod
    def _get_message_id(cls) -> int:
        return 2

    def _customize_sbe_dict(self, sbe_dict: dict) -> dict:
        """Flatten levels into individual bidPrice0-9, askPrice0-9, etc."""
        sbe_dict.pop('levels', None)

        for idx in range(10):
            sbe_dict.update(
                BidAskPair.to_dict(self.levels[idx] if idx < len(self.levels) else None, idx)
            )
        return sbe_dict

    @classmethod
    def _customize_from_message(cls, body: dict, kwargs: dict):
        """Reconstruct levels from flattened bidPrice0-9, askPrice0-9, etc."""
        kwargs['levels'] = [BidAskPair.from_dict(body, idx) for idx in range(10)]
        return cls(**kwargs)

@dataclass
class MBP1(SchemaBase, PriceMixin, SizeMixin):
    exchange_id: int
    security_id: int
    timestamp_event: int
    timestamp_sent: int | None
    timestamp_recv: int
    price: int | None
    size: int | None
    action: str
    side: str | None
    flags: list[str]
    sequence: int | None
    depth: int | None
    levels: list[BidAskPair]

    @classmethod
    def _get_message_id(cls) -> int:
        return 3

    def _customize_sbe_dict(self, sbe_dict: dict) -> dict:
        """Flatten single level into bidPrice0, askPrice0, etc."""
        sbe_dict.pop('levels', None)
        sbe_dict.update(
            BidAskPair.to_dict(self.levels[0] if self.levels else None, 0)
        )
        return sbe_dict

    @classmethod
    def _customize_from_message(cls, body: dict, kwargs: dict):
        """Reconstruct single level from flattened bidPrice0, askPrice0, etc."""
        kwargs['levels'] = [BidAskPair.from_dict(body, 0)]
        return cls(**kwargs)

@dataclass
class BBO(SchemaBase, PriceMixin, SizeMixin):
    exchange_id: int
    security_id: int
    timestamp_event: int
    timestamp_recv: int
    price: int | None
    size: int | None
    side: str | None
    flags: list[str]
    sequence: int | None
    levels: list[BidAskPair]

    _message_id: int = 4  # Default to BBO1S

    @classmethod
    def _get_message_id(cls) -> int:
        return getattr(cls, '_message_id', 4)

    def _customize_sbe_dict(self, sbe_dict: dict) -> dict:
        """Flatten single level into bidPrice0, askPrice0, etc."""
        sbe_dict.pop('levels', None)
        sbe_dict.update(
            BidAskPair.to_dict(self.levels[0] if self.levels else None, 0)
        )
        return sbe_dict

    @classmethod
    def _customize_from_message(cls, body: dict, kwargs: dict):
        """Reconstruct single level from flattened bidPrice0, askPrice0, etc."""
        kwargs['levels'] = [BidAskPair.from_dict(body, 0)]
        return cls(**kwargs)

class BBO1S(BBO):
    _message_id = 4

class BBO1M(BBO):
    _message_id = 5

@dataclass
class Trades(SchemaBase, PriceMixin, SizeMixin):
    exchange_id: int
    security_id: int
    timestamp_event: int
    timestamp_sent: int | None
    timestamp_recv: int
    price: int | None
    size: int | None
    action: str
    side: str | None
    flags: list[str]
    sequence: int | None
    depth: int | None

    @classmethod
    def _get_message_id(cls) -> int:
        return 6

@dataclass
class OHLCV(SchemaBase):
    exchange_id: int
    security_id: int
    timestamp_event: int
    open: int
    high: int
    low: int
    close: int
    volume: int

    _message_id: int = 7  # Default to OHLCV1S

    @classmethod
    def _get_message_id(cls) -> int:
        return getattr(cls, '_message_id', 7)

    @property
    def pretty_open(self):
        return self.open / FIXED_PRICE_SCALE

    @property
    def pretty_high(self):
        return self.high / FIXED_PRICE_SCALE

    @property
    def pretty_low(self):
        return self.low / FIXED_PRICE_SCALE

    @property
    def pretty_close(self):
        return self.close / FIXED_PRICE_SCALE

    @property
    def pretty_volume(self):
        return self.volume / FIXED_SIZE_SCALE

class OHLCV1S(OHLCV):
    _message_id = 7

class OHLCV1M(OHLCV):
    _message_id = 8

class OHLCV1H(OHLCV):
    _message_id = 9

def get_schema_base(schema_type: SchemaType) -> Type[SchemaBase]:
    if schema_type == SchemaType.MBP_10:
        return MBP10
    elif schema_type == SchemaType.MBP_1:
        return MBP1
    elif schema_type == SchemaType.BBO_1S:
        return BBO1S
    elif schema_type == SchemaType.BBO_1M:
        return BBO1M
    elif schema_type == SchemaType.TRADES:
        return Trades
    elif schema_type == SchemaType.OHLCV_1S:
        return OHLCV1S
    elif schema_type == SchemaType.OHLCV_1M:
        return OHLCV1M
    elif schema_type == SchemaType.OHLCV_1H:
        return OHLCV1H
    raise Exception(f"Schema type {schema_type} not implemented")
