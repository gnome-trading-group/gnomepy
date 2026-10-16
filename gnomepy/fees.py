"""Venue fees, as the registry stores them (sm.fee_schedule) and the Java simulator and gateways charge them.

A listing's fees over time are a :class:`FeeTimeline` of :class:`FeeSchedule`\\ s; load them with
:func:`load_fee_timelines`. Prices and fees are in price units (1e9 = $1) and quantities in size units (1e6 = 1
contract), as everywhere else, and every result matches gnome-registry's ``group.gnometrading.sm.fee.FeeSchedule``
exactly: both run the same floating-point steps and snap to 1e-12 dollars before rounding.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from datetime import datetime

from gnomepy.registry.api import RegistryClient

PRICE_SCALE = 1_000_000_000
SIZE_SCALE = 1_000_000
_PICO_PER_UNIT = 1_000
# Kalshi's model fee rounds up to $0.000001 before balance rounding.
_MICRO_DOLLAR_PICO = 1_000_000
OPEN_ENDED = 2**63 - 1

_KINDS = {"PRICE_CURVE", "NOTIONAL", "PER_UNIT"}
_SIDES = {"BOTH", "BUY", "SELL"}
_MODES = {"NONE", "CEIL", "HALF_UP", "HALF_EVEN"}
_SCOPES = {"FILL", "ORDER_ACCUMULATOR"}


@dataclass(frozen=True)
class FeeComponent:
    """One part of a fill's fee, charged at the taker or maker rate by the fill's liquidity; negative is a rebate."""

    kind: str
    taker: float
    maker: float
    exponent: float = 1.0
    side: str = "BOTH"


@dataclass
class OrderFeeState:
    """What one order carries from fill to fill for ``ORDER_ACCUMULATOR`` rounding."""

    accumulated_units: int = 0


@dataclass(frozen=True)
class FeeSchedule:
    """A venue's fee schedule for one listing over one stretch of time.

    ``rounding_increment_units`` is in price units; ``rounding_scope`` is ``FILL`` (each fill rounded on its own) or
    ``ORDER_ACCUMULATOR`` (Kalshi: the balance change is rounded and the excess carried across the order's fills).
    """

    currency: str
    components: tuple[FeeComponent, ...]
    rounding_mode: str = "NONE"
    rounding_increment_units: int = 0
    rounding_scope: str = "FILL"

    def __post_init__(self):
        if self.rounding_scope == "ORDER_ACCUMULATOR" and self.rounding_mode != "CEIL":
            raise ValueError("ORDER_ACCUMULATOR rounding must be CEIL")
        if self.rounding_mode != "NONE" and self.rounding_increment_units <= 0:
            raise ValueError("rounding needs a positive increment")

    @classmethod
    def from_definition(cls, definition: dict) -> FeeSchedule:
        """From the registry's JSON definition (``sm.fee_schedule.definition``)."""
        components = []
        for c in definition["components"]:
            kind, side = c["kind"], c.get("side", "BOTH")
            if kind not in _KINDS or side not in _SIDES:
                raise ValueError(f"unknown fee component {c}")
            components.append(FeeComponent(kind, float(c["taker"]), float(c["maker"]),
                                           float(c.get("exponent", 1.0)), side))
        rounding = definition["rounding"]
        mode = rounding["mode"]
        if mode not in _MODES:
            raise ValueError(f"unknown rounding mode {mode}")
        if mode == "NONE":
            return cls(definition["currency"], tuple(components))
        scope = rounding["scope"]
        if scope not in _SCOPES:
            raise ValueError(f"unknown rounding scope {scope}")
        increment = round(float(rounding["increment"]) * PRICE_SCALE)
        return cls(definition["currency"], tuple(components), mode, increment, scope)

    def fee(self, buy: bool, price: int, qty: int, maker: bool, state: OrderFeeState | None = None) -> int:
        """The fee for one fill in price units, rounded as the venue rounds it.

        ``state`` carries an order's rounding from fill to fill for ``ORDER_ACCUMULATOR``; pass the same instance for
        every fill of an order, or None to round the fill as if it were the order's only one.
        """
        pico = self.raw_pico(buy, price, qty, maker)
        if self.rounding_scope == "ORDER_ACCUMULATOR":
            return self._accumulate(buy, price, qty, pico, state)
        return self.round_fill(pico)

    @classmethod
    def from_java(cls, schedule) -> FeeSchedule:
        """From gnome-registry's ``group.gnometrading.sm.fee.FeeSchedule``."""
        components = tuple(
            FeeComponent(str(c.kind().name()), float(c.taker()), float(c.maker()), float(c.exponent()),
                         str(c.side().name()))
            for c in schedule.components()
        )
        r = schedule.rounding()
        return cls(str(schedule.currency()), components, str(r.mode().name()), int(r.incrementUnits()),
                   str(r.scope().name()))

    def price_curve_rates(self) -> tuple[float, float]:
        """(taker, maker) rates of a schedule that is one ``rate × p(1−p)`` curve on both sides, as Kalshi's and
        Polymarket's are, for code that solves for prices under that curve. Raises for any other shape."""
        if len(self.components) != 1:
            raise ValueError(f"not a single price curve: {self.components}")
        c = self.components[0]
        if c.kind != "PRICE_CURVE" or c.exponent != 1.0 or c.side != "BOTH":
            raise ValueError(f"not a p(1-p) curve on both sides: {c}")
        return c.taker, c.maker

    def fee_dollars(self, buy: bool, price: float, qty: float, maker: bool) -> float:
        """The fee in dollars for one fill of ``qty`` contracts at ``price`` dollars, as a lone fill."""
        return self.fee(buy, round(price * PRICE_SCALE), round(qty * SIZE_SCALE), maker) / PRICE_SCALE

    def raw_pico(self, buy: bool, price: int, qty: int, maker: bool) -> int:
        """The unrounded fee in picodollars (1e-3 price units), for fills summed before rounding once."""
        contracts = qty / SIZE_SCALE
        units = 0.0
        for c in self.components:
            if c.side != "BOTH" and (c.side == "BUY") != buy:
                continue
            rate = c.maker if maker else c.taker
            if rate == 0:
                continue
            if c.kind == "PRICE_CURVE":
                units += rate * contracts * _curve(price, c.exponent) * float(PRICE_SCALE)
            elif c.kind == "NOTIONAL":
                units += rate * contracts * price
            else:
                units += rate * contracts * float(PRICE_SCALE)
        return math.floor(units * float(_PICO_PER_UNIT) + 0.5)

    def round_fill(self, pico: int) -> int:
        """Rounds an unrounded fee (:meth:`raw_pico`) to price units as one fill."""
        if self.rounding_mode == "NONE":
            # Toward zero, as Java's integer division.
            return -((-pico) // _PICO_PER_UNIT) if pico < 0 else pico // _PICO_PER_UNIT
        increment = self.rounding_increment_units * _PICO_PER_UNIT
        if self.rounding_mode == "CEIL":
            steps = -((-pico) // increment)
        elif self.rounding_mode == "HALF_UP":
            steps = (abs(pico) + increment // 2) // increment
            steps = -steps if pico < 0 else steps
        else:
            quotient = pico // increment
            twice_remainder = 2 * (pico - quotient * increment)
            if twice_remainder > increment or (twice_remainder == increment and quotient % 2 != 0):
                quotient += 1
            steps = quotient
        return steps * self.rounding_increment_units

    def _accumulate(self, buy: bool, price: int, qty: int, pico: int, state: OrderFeeState | None) -> int:
        increment = self.rounding_increment_units
        trade_fee = -((-pico) // _MICRO_DOLLAR_PICO) * (_MICRO_DOLLAR_PICO // _PICO_PER_UNIT)
        notional = abs(price * qty) // SIZE_SCALE
        notional = -notional if (price < 0) != (qty < 0) else notional
        change = (-notional if buy else notional) - trade_fee
        rounding_fee = change - (change // increment) * increment
        gross = trade_fee + rounding_fee
        accumulated = (state.accumulated_units if state is not None else 0) + rounding_fee
        rebate = min((accumulated // increment) * increment, (gross // increment) * increment)
        if state is not None:
            state.accumulated_units = accumulated - rebate
        return gross - rebate


def _curve(price: int, exponent: float) -> float:
    p = price / float(PRICE_SCALE)
    shape = p * (1.0 - p)
    return shape if exponent == 1.0 else math.pow(shape, exponent)


def _epoch_nanos(value: str | datetime | int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    whole = int(value.timestamp())
    return whole * 1_000_000_000 + value.microsecond * 1_000


@dataclass
class FeeTimeline:
    """One listing's fee schedules over time: each holds from its start until its end (exclusive, epoch nanos)."""

    starts: list[int] = field(default_factory=list)
    ends: list[int] = field(default_factory=list)
    schedules: list[FeeSchedule] = field(default_factory=list)

    @classmethod
    def from_registry(cls, entries: list[dict]) -> FeeTimeline:
        """From one listing's ``timeline`` in ``GET /fees``."""
        timeline = cls()
        for e in entries:
            timeline.starts.append(_epoch_nanos(e["effectiveFrom"]))
            timeline.ends.append(OPEN_ENDED if e.get("effectiveTo") is None else _epoch_nanos(e["effectiveTo"]))
            timeline.schedules.append(FeeSchedule.from_definition(e["definition"]))
        return timeline

    def at(self, when: int | str | datetime) -> FeeSchedule | None:
        """The schedule in force at ``when`` (epoch nanos, ISO-8601 or datetime), or None when no fees are known."""
        nanos = _epoch_nanos(when)
        i = bisect.bisect_right(self.starts, nanos) - 1
        if i < 0 or nanos >= self.ends[i]:
            return None
        return self.schedules[i]


def load_fee_timelines(
    registry: RegistryClient,
    listing_ids: list[int],
    *,
    start: str | None = None,
    end: str | None = None,
) -> dict[int, FeeTimeline]:
    """Each listing's fee timeline from the registry, optionally trimmed to the schedules in force within
    [start, end]."""
    raw = registry.get_fee_timelines(listing_ids, start=start, end=end)
    return {listing_id: FeeTimeline.from_registry(entries) for listing_id, entries in raw.items()}
