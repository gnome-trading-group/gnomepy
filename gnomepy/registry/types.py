from dataclasses import dataclass
from enum import IntEnum


class SecurityType(IntEnum):
    SPOT = 0
    PERPETUAL = 1
    FUTURE = 2
    OPTION = 3
    EVENT_CONTRACT = 4


class ContractType(IntEnum):
    NONE = 0
    LINEAR_PERPETUAL = 1
    INVERSE_PERPETUAL = 2
    LINEAR_FUTURE = 3
    INVERSE_FUTURE = 4
    CALL_OPTION = 5
    PUT_OPTION = 6
    BINARY = 7
    MULTI_OUTCOME = 8


class AssetClass(IntEnum):
    CRYPTO = 0
    EQUITY = 1
    COMMODITY = 2
    FX = 3
    INDEX = 4
    PREDICTION = 5


@dataclass
class Exchange:
    exchange_id: int
    exchange_name: str
    region: str
    schema_type: str
    date_modified: str
    date_created: str


@dataclass
class Currency:
    currency_id: int
    symbol: str
    name: str | None
    decimals: int
    date_modified: str
    date_created: str


@dataclass
class Security:
    security_id: int
    symbol: str
    type: int
    contract_type: int
    asset_class: int
    base_currency_id: int | None
    quote_currency_id: int | None
    settle_currency_id: int | None
    inverse: bool
    is_quanto: bool
    expiry: str | None
    strike_price: int | None
    active: bool
    underlying_security_id: int | None
    description: str | None
    date_modified: str
    date_created: str
    base_currency: str | None = None
    quote_currency: str | None = None
    settle_currency: str | None = None


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
class ListingSpec:
    id: int
    listing_id: int
    tick_size: int
    lot_size: int
    min_notional: int
    contract_multiplier: int
    recorded_at: str


@dataclass
class Event:
    event_id: int
    title: str
    description: str | None
    category: str | None
    resolution_source: str | None
    tags: list[str] | None
    resolved: bool
    resolved_at: str | None
    expiry: str | None
    date_modified: str
    date_created: str


@dataclass
class EventContract:
    event_contract_id: int
    event_id: int
    security_id: int
    outcome_label: str
    date_created: str


@dataclass
class ContractRelationship:
    relationship_id: int
    security_id_a: int
    security_id_b: int
    relationship_type: str
    confidence: float
    method: str
    reviewed: bool
    reviewed_at: str | None
    date_created: str


@dataclass
class ExchangeEvent:
    exchange_event_id: int
    exchange_id: int
    event_id: int
    native_event_id: str
    raw_title: str
    date_created: str
