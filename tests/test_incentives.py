from gnomepy.incentives import IncentiveProgram, IncentiveTimeline, load_incentive_timelines

START = "2026-11-11T05:00:00.000Z"
END = "2026-11-12T05:00:00.000Z"


def _version(effective_from=START, period_end=END, reward="129.62", paid_out=False):
    return {
        "venueProgramId": "k1", "program": "LIQUIDITY", "venueMarket": "KXRT-VIO-60", "periodStart": START,
        "periodEnd": period_end, "reward": reward, "rewardPer": "PERIOD",
        "definition": {"targetSize": 1000, "discountFactor": 0.5}, "paidOut": paid_out, "effectiveFrom": effective_from,
    }


def test_a_version_reads_reward_in_price_units_and_its_venue_terms():
    program = IncentiveProgram.from_registry(_version())
    assert program.reward == 129_620_000_000
    assert program.definition["targetSize"] == 1000
    assert program.covers(program.period_start)
    assert not program.covers(program.period_end)


def test_a_program_pays_only_inside_the_version_in_effect():
    timeline = IncentiveTimeline.from_registry([{"venueProgramId": "k1", "versions": [_version()]}])
    assert [p.venue_program_id for p in timeline.in_force("2026-11-11T12:00:00Z")] == ["k1"]
    assert timeline.in_force("2026-11-11T04:59:59Z") == []
    assert timeline.in_force(END) == []


def test_a_cancellation_stops_the_program_from_when_it_took_effect():
    announced = _version(effective_from="2026-11-10T00:00:00.000Z")
    cancelled = _version(effective_from="2026-11-11T08:00:00.000Z", period_end="2026-11-11T08:00:00.000Z")
    timeline = IncentiveTimeline.from_registry([{"venueProgramId": "k1", "versions": [announced, cancelled]}])
    assert len(timeline.in_force("2026-11-11T07:00:00Z")) == 1
    assert timeline.in_force("2026-11-11T09:00:00Z") == []


def test_an_open_ended_program_pays_until_further_notice():
    timeline = IncentiveTimeline.from_registry([{"venueProgramId": "k1", "versions": [_version(period_end=None)]}])
    assert len(timeline.in_force("2030-01-01T00:00:00Z")) == 1


def test_load_incentive_timelines_parses_each_listing():
    class Registry:
        def get_incentive_timelines(self, listing_ids, start=None, end=None, known_as_of=None):
            assert known_as_of == "2026-11-01T00:00:00Z"
            return {7: [{"venueProgramId": "k1", "versions": [_version()]}], 9: []}

    timelines = load_incentive_timelines(Registry(), [7, 9], known_as_of="2026-11-01T00:00:00Z")
    assert len(timelines[7].programs) == 1
    assert timelines[9].programs == []
