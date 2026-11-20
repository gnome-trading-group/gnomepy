"""The prices a listing may trade at, as gnome-registry's Java ``PriceGrid`` defines them.

Both are checked against one set of cases (``tests/fixtures/price_grid_vectors.json``, kept identical to the
registry's ``price-grid-vectors.json``), so research rounds prices exactly as the OMS checks them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

OPEN = 2**63 - 1
"""The end of a grid with no upper bound (Java's ``Long.MAX_VALUE``)."""

MAX_BANDS = 16


@dataclass(frozen=True)
class PriceBand:
    """Prices from ``start`` up to ``end`` (None for open), stepping ``step``; in price units (1e9 = $1)."""

    start: int
    end: int | None
    step: int


class PriceGrid:
    """Contiguous bands from 0, each with its own step, up to an end (1e9 for a binary contract, open otherwise).

    Grid points are 0, every multiple of a band's step inside that band, and the end. A valid price is a grid point
    strictly between 0 and the end. ``floor``, ``ceil``, ``up`` and ``down`` answer grid points, which can be 0 or the
    end at the edges; bound them with ``min`` and ``max`` where a valid price is needed.
    """

    __slots__ = ("_starts", "_steps", "_end")

    def __init__(self, starts: tuple[int, ...], steps: tuple[int, ...], end: int):
        self._starts = starts
        self._steps = steps
        self._end = end

    @classmethod
    def single(cls, step: int, end: int = OPEN) -> PriceGrid:
        return cls.of([PriceBand(0, None if end == OPEN else end, step)])

    @classmethod
    def of(cls, bands: Sequence[PriceBand]) -> PriceGrid:
        """Raises ValueError unless the bands start at 0, follow on without gaps or overlaps, have positive steps,
        and every edge sits on the steps on both sides of it."""
        if not 0 < len(bands) <= MAX_BANDS:
            raise ValueError(f"A grid has 1 to {MAX_BANDS} bands, not {len(bands)}")
        if bands[0].start != 0:
            raise ValueError(f"A grid starts at 0, not {bands[0].start}")
        ends = [OPEN if band.end is None else band.end for band in bands]
        for i, band in enumerate(bands):
            if band.step <= 0:
                raise ValueError(f"Band {i} has step {band.step}")
            if ends[i] <= band.start:
                raise ValueError(f"Band {i} ends at {ends[i]}, not after {band.start}")
            if band.start % band.step != 0 or (ends[i] != OPEN and ends[i] % band.step != 0):
                raise ValueError(f"Band {i} [{band.start}, {ends[i]}) has an edge off its step {band.step}")
            if i + 1 < len(bands) and ends[i] != bands[i + 1].start:
                raise ValueError(f"Band {i} ends at {ends[i]} but the next starts at {bands[i + 1].start}")
        end = ends[-1]
        # A closed grid ends on every band's step, so mirroring it about its end lands every edge on a step.
        if end != OPEN:
            for band in bands:
                if end % band.step != 0:
                    raise ValueError(f"A grid ending at {end} must end on every band's step, not {band.step}")
        return cls(tuple(band.start for band in bands), tuple(band.step for band in bands), end)

    @classmethod
    def from_registry(cls, bands: Sequence[dict | PriceBand]) -> PriceGrid:
        """From a listing spec's ``price_bands`` as the registry serves them: ``{start, end, step}``, end null when
        open."""
        return cls.of([band if isinstance(band, PriceBand) else PriceBand(**band) for band in bands])

    @classmethod
    def from_java(cls, grid) -> PriceGrid:
        bands = int(grid.bands())
        return cls(
            tuple(int(grid.start(i)) for i in range(bands)),
            tuple(int(grid.step(i)) for i in range(bands)),
            int(grid.end()),
        )

    @property
    def end(self) -> int:
        return self._end

    @property
    def bands(self) -> list[PriceBand]:
        return [
            PriceBand(start, None if self._band_end(i) == OPEN else self._band_end(i), step)
            for i, (start, step) in enumerate(zip(self._starts, self._steps))
        ]

    def valid(self, price: int) -> bool:
        return 0 < price < self._end and price % self._steps[self._band(price)] == 0

    def floor(self, price: int) -> int:
        """The highest grid point at or below ``price``: 0 below the first step, the end at or past it."""
        if price <= 0:
            return 0
        if price >= self._end:
            return self._end
        return price - price % self._steps[self._band(price)]

    def ceil(self, price: int) -> int:
        """The lowest grid point at or above ``price``: 0 at or below 0, the end past the last band's last step."""
        if price <= 0:
            return 0
        if price >= self._end:
            return self._end
        step = self._steps[self._band(price)]
        below = price - price % step
        if below == price:
            return price
        return self._end if step >= self._end - below else below + step

    def up(self, price: int, steps: int = 1) -> int:
        """The grid point ``steps`` steps above ``price``, each step the one in force there; never past the end."""
        at = price
        for _ in range(steps):
            if at >= self._end:
                break
            at = self.ceil(at + 1)
        return at

    def down(self, price: int, steps: int = 1) -> int:
        """The grid point ``steps`` steps below ``price``; a step down from a band's start takes the band below's."""
        at = price
        for _ in range(steps):
            if at <= 0:
                break
            at = self.floor(at - 1)
        return at

    def ticks_between(self, a: int, b: int) -> int:
        """How many grid steps lie between two prices, counted from each one's floor and across bands."""
        return abs(self._index(b) - self._index(a))

    @property
    def min(self) -> int:
        """The lowest valid price."""
        return self._steps[0]

    @property
    def max(self) -> int:
        """The highest valid price."""
        last = self._steps[-1]
        return OPEN - OPEN % last if self._end == OPEN else self._end - last

    @property
    def finest_step(self) -> int:
        return min(self._steps)

    def complement(self) -> PriceGrid:
        """The grid in the complement's prices (``end - p``), as a binary contract's other side trades."""
        if self._end == OPEN:
            raise ValueError("A grid with no end has no complement")
        count = len(self._starts)
        return PriceGrid.of([
            PriceBand(self._end - self._band_end(i), self._end - self._starts[i], self._steps[i])
            for i in reversed(range(count))
        ])

    def _band_end(self, band: int) -> int:
        return self._starts[band + 1] if band + 1 < len(self._starts) else self._end

    def _band(self, price: int) -> int:
        band = len(self._starts) - 1
        while self._starts[band] > price:
            band -= 1
        return band

    def _index(self, price: int) -> int:
        at = self.floor(price)
        index = 0
        for i, start in enumerate(self._starts):
            band_end = self._band_end(i)
            if at < band_end or i + 1 == len(self._starts):
                return index + (at - start) // self._steps[i]
            index += (band_end - start) // self._steps[i]
        return index

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PriceGrid)
            and self._end == other._end
            and self._starts == other._starts
            and self._steps == other._steps
        )

    def __hash__(self) -> int:
        return hash((self._starts, self._steps, self._end))

    def __repr__(self) -> str:
        parts = []
        for i, (start, step) in enumerate(zip(self._starts, self._steps)):
            band_end = self._band_end(i)
            parts.append(f"[{start},{'open' if band_end == OPEN else band_end})@{step}")
        return f"PriceGrid({' '.join(parts)})"
