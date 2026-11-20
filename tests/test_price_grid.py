"""The Python grid against the registry's shared cases, so research rounds prices as the OMS checks them."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gnomepy.price_grid import OPEN, PriceBand, PriceGrid
from gnomepy.registry.types import ListingSpec

VECTORS = json.loads((Path(__file__).parent / "fixtures" / "price_grid_vectors.json").read_text())


def _grid(bands) -> PriceGrid:
    return PriceGrid.of([PriceBand(start, end, step) for start, end, step in bands])


GRIDS = {name: _grid(bands) for name, bands in VECTORS["grids"].items()}


def _answer(grid: PriceGrid, op: str, args: list):
    if op == "valid":
        return grid.valid(*args)
    if op == "floor":
        return grid.floor(*args)
    if op == "ceil":
        return grid.ceil(*args)
    if op == "up":
        return grid.up(*args)
    if op == "down":
        return grid.down(*args)
    if op == "ticksBetween":
        return grid.ticks_between(*args)
    if op == "min":
        return grid.min
    if op == "max":
        return grid.max
    if op == "finestStep":
        return grid.finest_step
    if op == "bands":
        return len(grid.bands)
    if op == "complement":
        return grid.complement()
    raise AssertionError(f"no such op {op}")


@pytest.mark.parametrize("case", VECTORS["cases"], ids=lambda c: f"{c['grid']} {c['op']} {c['args']}")
def test_shared_cases(case):
    grid = GRIDS[case["grid"]]
    expected = GRIDS[case["expected"]] if case["op"] == "complement" else case["expected"]
    assert _answer(grid, case["op"], case["args"]) == expected


@pytest.mark.parametrize("case", VECTORS["invalid"], ids=lambda c: c["name"])
def test_invalid_grids_are_refused(case):
    with pytest.raises(ValueError):
        _grid(case["bands"])


def test_an_open_grid_has_no_complement():
    with pytest.raises(ValueError):
        PriceGrid.single(1).complement()


def test_the_vectors_match_the_registrys():
    registry = Path(__file__).parents[2] / "gnome-registry" / "src" / "test" / "resources" / "price-grid-vectors.json"
    if not registry.exists():
        pytest.skip("no gnome-registry checkout beside this one")
    assert registry.read_bytes() == (Path(__file__).parent / "fixtures" / "price_grid_vectors.json").read_bytes()


def test_a_spec_from_the_registry_carries_its_grid():
    spec = ListingSpec(
        id=1,
        listing_id=7,
        lot_size=10_000,
        min_notional=0,
        contract_multiplier=1_000_000_000,
        recorded_at="2026-11-18T00:00:00Z",
        price_bands=[
            {"start": 0, "end": 100_000_000, "step": 1_000_000},
            {"start": 100_000_000, "end": 900_000_000, "step": 10_000_000},
            {"start": 900_000_000, "end": 1_000_000_000, "step": 1_000_000},
        ],
    )

    assert spec.grid == GRIDS["tapered_deci_cent"]
    assert spec.price_bands[0] == PriceBand(0, 100_000_000, 1_000_000)


def test_an_open_band_from_the_registry_has_no_end():
    grid = PriceGrid.from_registry([{"start": 0, "end": None, "step": 1}])

    assert grid.end == OPEN
    assert grid.bands == [PriceBand(0, None, 1)]
