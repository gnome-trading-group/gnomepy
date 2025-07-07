from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntFlag, StrEnum
from typing import Type
from gnomepy.data.sbe import DecodedMessage
from enum import Enum

FIXED_PRICE_SCALE = 1e9
FIXED_SIZE_SCALE = 1e6

@dataclass
class Listing:
    """A class representing a security listing on an exchange.
    
    Attributes:
        exchange_id (int): The exchange identifier where the security is listed
        security_id (int): The security identifier
    """
    exchange_id: int
    security_id: int
    
    def __hash__(self) -> int:
        """Make the Listing object hashable for use as dictionary keys."""
        return hash((self.exchange_id, self.security_id))
    
    def __eq__(self, other) -> bool:
        """Define equality for Listing objects."""
        if not isinstance(other, Listing):
            return False
        return (self.exchange_id, self.security_id) == (other.exchange_id, other.security_id)
    
    def __str__(self) -> str:
        """String representation: exchange_id_security_id."""
        return f"{self.exchange_id}_{self.security_id}"

class Action(Enum):
    BUY = "BUY"
    SELL = "SELL" 
    NEUTRAL = "NEUTRAL"

class Status(Enum):
    OPEN = "OPEN" 
    FILLED = "FILLED"

class OrderType(Enum):
    MARKET = "MARKET" 
    LIMIT = "LIMIT"
    # TODO: stop or stop loss or something else here

class Order:
    listing: Listing
    size: float
    status: Status
    type: OrderType
    action: Action
    price: float
    cash_size: float
    timestampOpened: int
    timestampClosed: int
    signal: 'Signal'  # Forward reference since Signal is defined later

    def __init__(self, listing: Listing, size: float, status: Status, type: OrderType, 
                 action: Action, price: float, cash_size: float, timestampOpened: int = None,
                 signal: 'Signal' = None):
        self.listing = listing
        self.size = size
        self.status = status
        self.type = type
        self.action = action
        self.price = price
        self.cash_size = cash_size
        self.timestampOpened = timestampOpened
        self.timestampClosed = None
        self.signal = signal
        
    def close(self, timestampClosed: int):
        """Update the timestampClosed and status when order is closed"""
        self.timestampClosed = timestampClosed
        self.status = Status.FILLED
class SignalType(StrEnum):
    """A class representing the type of signal being generated.
    
    Values:
        ENTER_NEGATIVE_MEAN_REVERSION: Signal to enter a negative mean reversion trade
        ENTER_POSITIVE_MEAN_REVERSION: Signal to enter a positive mean reversion trade  
        EXIT_NEGATIVE_MEAN_REVERSION: Signal to exit a negative mean reversion trade
        EXIT_POSITIVE_MEAN_REVERSION: Signal to exit a positive mean reversion trade
    """
    ENTER_NEGATIVE_MEAN_REVERSION = "enter_negative_mean_reversion"
    ENTER_POSITIVE_MEAN_REVERSION = "enter_positive_mean_reversion"
    EXIT_NEGATIVE_MEAN_REVERSION = "exit_negative_mean_reversion" 
    EXIT_POSITIVE_MEAN_REVERSION = "exit_positive_mean_reversion"


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

    @property
    def pretty_bid_px(self) -> float:
        return self.bid_px / FIXED_PRICE_SCALE

    @property
    def pretty_ask_px(self) -> float:
        return self.ask_px / FIXED_PRICE_SCALE

class SchemaBase(ABC):
    @classmethod
    @abstractmethod
    def from_message(cls, message: DecodedMessage):
        raise NotImplemented

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
    def from_message(cls, message: DecodedMessage):
        body = message.value
        return cls(
            body['exchangeId'],
            body['securityId'],
            body['timestampEvent'],
            body['timestampSent'],
            body['timestampRecv'],
            body['price'],
            body['size'],
            body['action'],
            body['side'],
            body['flags'],
            body['sequence'],
            body['depth'],
            [BidAskPair.from_dict(body, idx) for idx in range(10)]
        )

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
    levels: BidAskPair

    @classmethod
    def from_message(cls, message: DecodedMessage):
        body = message.value
        return cls(
            body['exchangeId'],
            body['securityId'],
            body['timestampEvent'],
            body['timestampSent'],
            body['timestampRecv'],
            body['price'],
            body['size'],
            body['action'],
            body['side'],
            body['flags'],
            body['sequence'],
            body['depth'],
            BidAskPair.from_dict(body, 0)
        )

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
    levels: BidAskPair

    @classmethod
    def from_message(cls, message: DecodedMessage):
        body = message.value
        return cls(
            body['exchangeId'],
            body['securityId'],
            body['timestampEvent'],
            body['timestampRecv'],
            body['price'],
            body['size'],
            body['side'],
            body['flags'],
            body['sequence'],
            BidAskPair.from_dict(body, 0)
        )

BBO1S = BBO
BBO1M = BBO

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
    def from_message(cls, message: DecodedMessage):
        body = message.value
        return cls(
            body['exchangeId'],
            body['securityId'],
            body['timestampEvent'],
            body['timestampSent'],
            body['timestampRecv'],
            body['price'],
            body['size'],
            body['action'],
            body['side'],
            body['flags'],
            body['sequence'],
            body['depth'],
        )



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

    @classmethod
    def from_message(cls, message: DecodedMessage):
        body = message.value
        return cls(
            body['exchangeId'],
            body['securityId'],
            body['timestampEvent'],
            body['open'],
            body['high'],
            body['low'],
            body['close'],
            body['volume'],
        )

OHLCV1S = OHLCV
OHLCV1M = OHLCV
OHLCV1H = OHLCV

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

