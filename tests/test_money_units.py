"""Money values from Java are in price units (1e9 = $1) — no JVM needed."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gnomepy.java.enums import Side
from gnomepy.java.oms import PositionViewWrapper
from gnomepy.java.recorder import BacktestResults
from gnomepy.java.statics import Scales
from gnomepy.price_grid import PriceBand, PriceGrid

PRICE_SCALE = 1_000_000_000
SIZE_SCALE = 1_000_000


@pytest.fixture(autouse=True)
def _scales(monkeypatch):
    monkeypatch.setattr(Scales, "PRICE", PRICE_SCALE)
    monkeypatch.setattr(Scales, "SIZE", SIZE_SCALE)


class _Column:
    def __init__(self, name, col_type, idx):
        self._name, self._type, self._idx = name, col_type, idx

    def name(self):
        return self._name

    def type(self):
        return self._type

    def columnIndex(self):
        return self._idx


class _FakeBuffer:
    def __init__(self, columns: dict[str, tuple[str, list]]):
        self._columns = [_Column(name, t, i) for i, (name, (t, _)) in enumerate(columns.items())]
        self._values = [vals for _, vals in columns.values()]
        self._n = len(self._values[0])

    def getCount(self):
        return self._n

    def getColumns(self):
        return self._columns

    def getLongColumn(self, idx):
        return self._values[idx]

    def getByteColumn(self, idx):
        return self._values[idx]


class _FakeRecorder:
    def __init__(self, buffer):
        self._buffer = buffer

    def getOrderRecords(self):
        return self._buffer

    def getOrderRecordCount(self):
        return self._buffer.getCount()


def _orders_results(filled_qty: list[int], total_cost: list[int]) -> BacktestResults:
    n = len(filled_qty)
    buffer = _FakeBuffer({
        "submit_timestamp": ("LONG", list(range(1, n + 1))),
        "ack_timestamp": ("LONG", [0] * n),
        "terminal_timestamp": ("LONG", [0] * n),
        "side": ("BYTE", [1] * n),
        "order_type": ("BYTE", [0] * n),
        "final_status": ("BYTE", [0] * n),
        "submit_price": ("LONG", [0] * n),
        "submit_size": ("LONG", filled_qty),
        "filled_qty": ("LONG", filled_qty),
        "leaves_qty": ("LONG", [0] * n),
        "total_cost": ("LONG", total_cost),
    })
    return BacktestResults(_FakeRecorder(buffer))


class TestOrdersAvgFillPrice:
    def test_single_fill_recovers_price(self):
        # 2.5 units @ $100.25 -> notional $250.625 in price units
        qty = 2_500_000
        price = 100_250_000_000
        total_cost = price * qty // SIZE_SCALE
        df = _orders_results([qty], [total_cost]).orders_df(scale_prices=False)
        assert df["avg_fill_price"].iloc[0] == price
        assert df["avg_fill_price"].dtype == np.int64

    def test_vwap_rounds_not_truncates(self):
        # 1 unit @ 1.000000001 + 2 units @ 1.000000002 -> 1.0000000016666...
        total_cost = (1_000_000_001 * 1_000_000 + 1_000_000_002 * 2_000_000) // SIZE_SCALE
        df = _orders_results([3_000_000], [total_cost]).orders_df(scale_prices=False)
        assert df["avg_fill_price"].iloc[0] == 1_000_000_002

    def test_unfilled_is_zero(self):
        df = _orders_results([0], [0]).orders_df(scale_prices=False)
        assert df["avg_fill_price"].iloc[0] == 0

    def test_scaled_to_dollars(self):
        qty = 1_000_000
        price = 50_000_500_000_000
        df = _orders_results([qty], [price]).orders_df()
        assert df["avg_fill_price"].iloc[0] == pytest.approx(50_000.5)


class _JavaGrid:
    """Stands in for a Java PriceGrid: the same accessors, answered from the Python grid."""

    def __init__(self, grid: PriceGrid):
        self._bands = grid.bands
        self._end = grid.end

    def bands(self):
        return len(self._bands)

    def start(self, band):
        return self._bands[band].start

    def step(self, band):
        return self._bands[band].step

    def end(self):
        return self._end


class _FakeSecurityMaster:
    def __init__(self, lot_size: int, min_notional: int, min_size: int, grid: PriceGrid | None = None):
        java_grid = _JavaGrid(grid or PriceGrid.single(1))
        self._spec = SimpleNamespace(
            lotSize=lambda: lot_size,
            minNotional=lambda: min_notional,
            minSize=lambda: min_size,
            grid=lambda: java_grid,
        )

    def getListing(self, exchange_id, security_id):
        return SimpleNamespace(listingId=lambda: 1)

    def getListingSpec(self, listing_id):
        return self._spec


def _wrapper(lot_size: int, min_notional: int, min_size: int = 0) -> PositionViewWrapper:
    return PositionViewWrapper(None, _FakeSecurityMaster(lot_size, min_notional, min_size))


class TestCompliantSize:
    def test_min_notional_in_price_units(self):
        # $5 min notional at $0.40 -> 12.5 units, rounded up to whole lots of 1 unit = 13
        pv = _wrapper(lot_size=SIZE_SCALE, min_notional=5 * PRICE_SCALE)
        size = pv.compliant_size(1, 1, desired_size=SIZE_SCALE, price=400_000_000)
        assert size == 13 * SIZE_SCALE

    def test_min_notional_ceil_without_lot(self):
        # $10 at $3 -> 3.333334 units (ceil at size-unit resolution)
        pv = _wrapper(lot_size=0, min_notional=10 * PRICE_SCALE)
        size = pv.compliant_size(1, 1, desired_size=1, price=3 * PRICE_SCALE)
        assert size == 3_333_334

    def test_desired_above_min_unchanged(self):
        pv = _wrapper(lot_size=SIZE_SCALE // 100, min_notional=5 * PRICE_SCALE)
        size = pv.compliant_size(1, 1, desired_size=20 * SIZE_SCALE, price=PRICE_SCALE)
        assert size == 20 * SIZE_SCALE

    def test_lot_rounding_only(self):
        pv = _wrapper(lot_size=SIZE_SCALE, min_notional=0)
        size = pv.compliant_size(1, 1, desired_size=1_500_000, price=PRICE_SCALE)
        assert size == 2 * SIZE_SCALE

    def test_min_size_raises_small_order(self):
        # Polymarket: 5-share minimum, 0.01-share lots
        pv = _wrapper(lot_size=SIZE_SCALE // 100, min_notional=0, min_size=5 * SIZE_SCALE)
        size = pv.compliant_size(1, 1, desired_size=2 * SIZE_SCALE, price=PRICE_SCALE // 10)
        assert size == 5 * SIZE_SCALE

    def test_min_size_rounded_up_to_lot(self):
        pv = _wrapper(lot_size=2 * SIZE_SCALE, min_notional=0, min_size=5 * SIZE_SCALE)
        size = pv.compliant_size(1, 1, desired_size=1, price=PRICE_SCALE)
        assert size == 6 * SIZE_SCALE


class TestCompliantPrice:
    CENT = PRICE_SCALE // 100
    MILLI = PRICE_SCALE // 1000
    TAPERED = PriceGrid.of([
        PriceBand(0, 100_000_000, 1_000_000),
        PriceBand(100_000_000, 900_000_000, 10_000_000),
        PriceBand(900_000_000, 1_000_000_000, 1_000_000),
    ])

    def _pv(self, grid: PriceGrid | None) -> PositionViewWrapper:
        return PositionViewWrapper(None, _FakeSecurityMaster(0, 0, 0, grid=grid))

    def test_bids_round_down_and_asks_round_up(self):
        pv = self._pv(PriceGrid.single(self.CENT, PRICE_SCALE))
        off_tick = 41 * self.CENT + self.CENT // 2  # $0.415
        assert pv.compliant_price(1, 1, off_tick, Side.BID) == 41 * self.CENT
        assert pv.compliant_price(1, 1, off_tick, Side.ASK) == 42 * self.CENT

    def test_on_grid_prices_are_unchanged(self):
        pv = self._pv(PriceGrid.single(self.CENT, PRICE_SCALE))
        assert pv.compliant_price(1, 1, 37 * self.CENT, Side.BID) == 37 * self.CENT
        assert pv.compliant_price(1, 1, 37 * self.CENT, Side.ASK) == 37 * self.CENT

    def test_a_tapered_grid_keeps_tenths_of_a_cent_only_in_its_tails(self):
        pv = self._pv(self.TAPERED)
        assert pv.compliant_price(1, 1, 47 * self.MILLI, Side.BID) == 47 * self.MILLI
        assert pv.compliant_price(1, 1, 475 * self.MILLI, Side.BID) == 47 * self.CENT
        assert pv.compliant_price(1, 1, 475 * self.MILLI, Side.ASK) == 48 * self.CENT
        assert pv.compliant_price(1, 1, 995 * self.MILLI, Side.ASK) == 995 * self.MILLI

    def test_the_grid_is_its_listing_specs(self):
        assert self._pv(self.TAPERED).price_grid(1, 1) == self.TAPERED

    def test_no_spec_leaves_the_price(self):
        class _NoSpecs(_FakeSecurityMaster):
            def getListingSpec(self, listing_id):
                return None

        pv = PositionViewWrapper(None, _NoSpecs(0, 0, 0))
        assert pv.price_grid(1, 1) is None
        assert pv.compliant_price(1, 1, 412_345_678, Side.ASK) == 412_345_678


class TestSpecsAtTheBacktestsTime:
    def test_a_clock_selects_the_spec_in_force_at_its_time(self):
        class _DatedSecurityMaster(_FakeSecurityMaster):
            def getListingSpec(self, listing_id, nanos=None):
                tick = PRICE_SCALE // 100 if nanos is not None and nanos >= 1_000 else PRICE_SCALE // 10
                grid = _JavaGrid(PriceGrid.single(tick, PRICE_SCALE))
                return SimpleNamespace(grid=lambda: grid)

        clock = SimpleNamespace(now=0)
        clock.nanoTime = lambda: clock.now
        pv = PositionViewWrapper(None, _DatedSecurityMaster(0, 0, 0), clock)

        assert pv.compliant_price(1, 1, 415_000_000, Side.BID) == 400_000_000
        clock.now = 1_000
        assert pv.compliant_price(1, 1, 415_000_000, Side.BID) == 410_000_000
