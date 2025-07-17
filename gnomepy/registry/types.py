from dataclasses import dataclass
from enum import IntEnum


class SecurityType(IntEnum):
    SPOT = 0
    PERPETUAL = 1

@dataclass
class Listing:
    listing_id: int
    security_id: int
    exchange_id: int
    exchange_security_id: str | None
    exchange_security_symbol: str | None
    date_modified: str
    date_created: str

@dataclass
class Security:
    security_id: int
    symbol: str
    type: SecurityType
    description: str
    date_modified: str
    date_created: str

@dataclass
class Exchange:
    exchange_id: int
    exchange_name: str
    date_modified: str
    date_created: str
