import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gnomepy.fees import OPEN_ENDED, FeeSchedule, FeeTimeline, OrderFeeState, load_fee_timelines
from gnomepy.java.oms import PositionViewWrapper

# The same cases gnome-registry's FeeVectorsTest runs against the Java calculator.
VECTORS = json.loads((Path(__file__).parent / "fixtures" / "fee_vectors.json").read_text())["cases"]


@pytest.mark.parametrize("case", VECTORS, ids=[c["name"] for c in VECTORS])
def test_reproduces_the_shared_cases(case):
    schedule = FeeSchedule.from_definition(case["definition"])
    order = OrderFeeState()
    for fill in case["fills"]:
        fee = schedule.fee(fill["buy"], fill["price"], fill["qty"], fill["maker"],
                           order if case["sharedOrderState"] else None)
        assert fee == fill["fee"], fill


def test_fee_in_dollars_for_research():
    schedule = FeeSchedule.from_definition(VECTORS[4]["definition"])
    assert schedule.fee_dollars(True, 0.436, 1, False) == 0.02


def _entry(start, end, taker):
    return {
        "effectiveFrom": start, "effectiveTo": end, "scope": "EXCHANGE", "feeScheduleId": "1",
        "definition": {"currency": "USD", "components": [{"kind": "PRICE_CURVE", "taker": taker, "maker": 0}],
                       "rounding": {"mode": "NONE"}},
    }


def test_timeline_holds_each_schedule_until_the_next_and_leaves_gaps_unknown():
    timeline = FeeTimeline.from_registry([
        _entry("2026-04-03T04:00:00.000Z", "2026-07-01T04:00:00.000Z", 0.05),
        _entry("2026-07-01T04:00:00.000Z", None, 0.06),
    ])
    assert timeline.at("2026-04-03T03:59:59Z") is None
    assert timeline.at("2026-04-03T04:00:00Z").components[0].taker == 0.05
    assert timeline.at("2026-07-01T04:00:00Z").components[0].taker == 0.06
    assert timeline.ends[-1] == OPEN_ENDED


def test_load_fee_timelines_parses_each_listing():
    class Registry:
        def get_fee_timelines(self, listing_ids, start=None, end=None):
            return {7: [_entry("2026-01-01T00:00:00Z", None, 0.07)], 9: []}

    timelines = load_fee_timelines(Registry(), [7, 9])
    assert timelines[7].at("2026-06-01T00:00:00Z").components[0].taker == 0.07
    assert timelines[9].at("2026-06-01T00:00:00Z") is None


def test_an_accumulator_must_round_up():
    with pytest.raises(ValueError):
        FeeSchedule.from_definition({
            "currency": "USD", "components": [{"kind": "PRICE_CURVE", "taker": 0.07, "maker": 0}],
            "rounding": {"mode": "HALF_UP", "increment": 0.0001, "scope": "ORDER_ACCUMULATOR"},
        })


def _java_enum(name):
    return SimpleNamespace(name=lambda: name)


def _java_schedule(taker, maker):
    component = SimpleNamespace(kind=lambda: _java_enum("PRICE_CURVE"), taker=lambda: taker, maker=lambda: maker,
                                exponent=lambda: 1.0, side=lambda: _java_enum("BOTH"))
    rounding = SimpleNamespace(mode=lambda: _java_enum("CEIL"), incrementUnits=lambda: 100_000,
                               scope=lambda: _java_enum("ORDER_ACCUMULATOR"))
    # A class rather than a namespace: like JPype's proxies, a Java object is hashable.
    class JavaFeeSchedule:
        def components(self):
            return [component]

        def rounding(self):
            return rounding

        def currency(self):
            return "USD"

    return JavaFeeSchedule()


def test_strategies_get_the_schedule_in_force_at_the_backtests_time():
    july, august = _java_schedule(0.07, 0.0175), _java_schedule(0.035, 0.00875)

    class Timeline:
        def at(self, nanos):
            return None if nanos < 100 else (july if nanos < 200 else august)

    security_master = SimpleNamespace(getListing=lambda e, s: SimpleNamespace(listingId=lambda: 5),
                                      getFeeTimeline=lambda listing_id: Timeline())
    clock = SimpleNamespace(now=50)
    clock.nanoTime = lambda: clock.now
    positions = PositionViewWrapper(None, security_master, clock)

    assert positions.fee_schedule(1, 2) is None
    clock.now = 150
    assert positions.fee_schedule(1, 2).price_curve_rates() == (0.07, 0.0175)
    assert positions.fee_schedule(1, 2) is positions.fee_schedule(1, 2)
    clock.now = 250
    schedule = positions.fee_schedule(1, 2)
    assert schedule.price_curve_rates() == (0.035, 0.00875)
    assert schedule.fee(True, 550_000_000, 1_000_000, False) == 8_700_000


def test_price_curve_rates_refuse_other_shapes():
    with pytest.raises(ValueError):
        FeeSchedule.from_definition(VECTORS[-1]["definition"]).price_curve_rates()
