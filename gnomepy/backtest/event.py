from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from gnomepy.data.types import SchemaBase, MBP10, MBP1, BBO1S, BBO1M, Trades, OHLCV1H, OHLCV1M, \
    OrderExecutionReport, OHLCV1S, Order, CancelOrder, MBO

LocalMessage = Order | CancelOrder
ExchangeMessage = OrderExecutionReport


class EventType(IntEnum):
    EXCHANGE_MARKET_DATA = 0 # when a simulated exchange processes a market update (ie, fills trades)
    EXCHANGE_MESSAGE = 1 # messages sent from the exchange to local strat (execution reports)
    LOCAL_MARKET_DATA = 2 # when the local strategy processes a market update
    LOCAL_MESSAGE = 3 # messages sent from the strategy to the exchange (orders, cancels, etc)

@dataclass
class Event:
    timestamp: int
    event_type: EventType
    data: Any

    @classmethod
    def from_schema_local(cls, schema: SchemaBase):
        if isinstance(schema, (MBO, MBP10, MBP1, Trades)):
            timestamp = schema.timestamp_recv
        elif isinstance(schema, (BBO1S, BBO1M, OHLCV1H, OHLCV1M, OHLCV1S)):
            timestamp = schema.timestamp_event
        else:
            raise ValueError(f"Schema class {type(schema)} is not implemented for local data")
        return cls(timestamp, EventType.LOCAL_MARKET_DATA, schema)

    @classmethod
    def from_schema_exchange(cls, schema: SchemaBase):
        if isinstance(schema, (MBO, MBP10, MBP1, Trades, BBO1M, BBO1S, OHLCV1S, OHLCV1M, OHLCV1H)):
            timestamp = schema.timestamp_event
        else:
            raise ValueError(f"Schema class {type(schema)} is not implemented for exchange data")
        return cls(timestamp, EventType.EXCHANGE_MARKET_DATA, schema)

    @classmethod
    def from_local_message(cls, message: LocalMessage, timestamp: int):
        return cls(timestamp, EventType.LOCAL_MESSAGE, message)

    @classmethod
    def from_exchange_message(cls, exchange_message: ExchangeMessage, timestamp: int):
        return cls(timestamp, EventType.EXCHANGE_MESSAGE, exchange_message)

    def __lt__(self, other):
        return self.timestamp < other.timestamp
