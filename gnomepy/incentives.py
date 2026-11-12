"""Venue incentive programs (liquidity rewards), as the registry stores them (sm.incentive_program).

A listing's programs are an :class:`IncentiveTimeline`: each program as its versions over time. At time t a program has
the version with the latest start at or before t, and pays when t falls in that version's period, so a later
cancellation (a version whose period ends before it starts) stops it from then on. Load them with
:func:`load_incentive_timelines`. Rewards are in price units (1e9 = $1), as everywhere else; ``definition`` holds the
venue's own scoring terms.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_DOWN, Decimal

from gnomepy.fees import OPEN_ENDED, PRICE_SCALE, _epoch_nanos
from gnomepy.registry.api import RegistryClient


@dataclass(frozen=True)
class IncentiveProgram:
    """One version of a program: the terms that held from ``effective_from`` until the program's next version."""

    venue_program_id: str
    program: str
    venue_market: str
    period_start: int
    period_end: int
    reward: int
    reward_per: str
    definition: dict
    paid_out: bool | None
    effective_from: int

    @classmethod
    def from_registry(cls, version: dict) -> IncentiveProgram:
        """From one entry of a program's ``versions`` in ``GET /incentive-programs``."""
        reward = (Decimal(version["reward"]) * PRICE_SCALE).quantize(Decimal(1), rounding=ROUND_DOWN)
        return cls(
            venue_program_id=version["venueProgramId"],
            program=version["program"],
            venue_market=version["venueMarket"],
            period_start=_epoch_nanos(version["periodStart"]),
            period_end=OPEN_ENDED if version.get("periodEnd") is None else _epoch_nanos(version["periodEnd"]),
            reward=int(reward),
            reward_per=version["rewardPer"],
            definition=version["definition"],
            paid_out=version.get("paidOut"),
            effective_from=_epoch_nanos(version["effectiveFrom"]),
        )

    def covers(self, nanos: int) -> bool:
        """True when ``nanos`` falls in the program's period; a cancelled program's period is empty."""
        return self.period_start <= nanos < self.period_end


@dataclass
class ProgramVersions:
    """One program's versions, in effective order."""

    venue_program_id: str
    versions: list[IncentiveProgram] = field(default_factory=list)

    def version_at(self, when: int | str | datetime) -> IncentiveProgram | None:
        """The version in effect at ``when``, or None before the first."""
        nanos = _epoch_nanos(when)
        i = bisect.bisect_right([v.effective_from for v in self.versions], nanos) - 1
        return self.versions[i] if i >= 0 else None

    def in_force(self, when: int | str | datetime) -> IncentiveProgram | None:
        """The program as it pays at ``when``, or None when it isn't paying then."""
        version = self.version_at(when)
        return version if version is not None and version.covers(_epoch_nanos(when)) else None


@dataclass
class IncentiveTimeline:
    """One listing's incentive programs over time."""

    programs: list[ProgramVersions] = field(default_factory=list)

    @classmethod
    def from_registry(cls, programs: list[dict]) -> IncentiveTimeline:
        """From one listing's ``programs`` in ``GET /incentive-programs``."""
        return cls([
            ProgramVersions(p["venueProgramId"], [IncentiveProgram.from_registry(v) for v in p["versions"]])
            for p in programs
        ])

    def in_force(self, when: int | str | datetime) -> list[IncentiveProgram]:
        """Every program paying at ``when``."""
        return [v for v in (p.in_force(when) for p in self.programs) if v is not None]


def load_incentive_timelines(
    registry: RegistryClient,
    listing_ids: list[int],
    *,
    start: str | None = None,
    end: str | None = None,
    known_as_of: str | None = None,
) -> dict[int, IncentiveTimeline]:
    """Each listing's incentive programs from the registry, optionally limited to those that could pay within
    [start, end) and to what the registry knew at ``known_as_of``."""
    raw = registry.get_incentive_timelines(listing_ids, start=start, end=end, known_as_of=known_as_of)
    return {listing_id: IncentiveTimeline.from_registry(programs) for listing_id, programs in raw.items()}
