from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from gnomepy import SchemaBase, MBP10, MBP1, BBO1S, BBO1M, Trades, OHLCV1H, OHLCV1M, Order, \
    OrderExecutionReport, OHLCV1S, LocalMessage


class EventType(IntEnum):
    MARKET_DATA = 0
    LOCAL_MESSAGE = 1
    EXECUTION_REPORT = 2

@dataclass
class Event:
    timestamp: int
    event_type: EventType
    data: Any

    @classmethod
    def from_schema(cls, schema: SchemaBase):
        if isinstance(schema, (MBP10, MBP1, Trades)):
            timestamp = schema.timestamp_recv
        elif isinstance(schema, (BBO1S, BBO1M, OHLCV1H, OHLCV1M, OHLCV1S)):
            timestamp = schema.timestamp_event
        else:
            raise ValueError(f"Schema class {type(schema)} is not implemented")
        return cls(timestamp, EventType.MARKET_DATA, schema)

    @classmethod
    def from_local_message(cls, message: LocalMessage, timestamp: int):
        return cls(timestamp, EventType.LOCAL_MESSAGE, message)

    @classmethod
    def from_execution_report(cls, execution_report: OrderExecutionReport, timestamp: int):
        return cls(timestamp, EventType.EXECUTION_REPORT, execution_report)

    def __lt__(self, other):
        return self.timestamp < other.timestamp
