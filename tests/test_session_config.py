from __future__ import annotations

from gnomepy.java.strategy.config import (
    LogNormalLatencyConfig,
    ListingSessionConfig,
    SessionConfig,
    SimulationProfile,
)


def _paper(**kwargs) -> SessionConfig:
    return SessionConfig(
        mode="paper",
        listings=[ListingSessionConfig(listing_id=7, profile="default")],
        profiles={"default": SimulationProfile(network_latency=LogNormalLatencyConfig(seed=3))},
        **kwargs,
    )


def test_seed_omitted_by_default():
    assert "simulation.seed" not in _paper().to_properties()


def test_seed_written_for_paper_sessions():
    props = _paper(seed=99).to_properties()
    assert props["simulation.seed"] == "99"
    assert props["simulation.profiles.default.network.latency.seed"] == "3"


def test_seed_parsed_from_yaml(tmp_path):
    path = tmp_path / "session.yaml"
    path.write_text(
        "mode: paper\n"
        "seed: 42\n"
        "listings:\n  - listing_id: 7\n    profile: default\n"
        "profiles:\n  default:\n    network_latency:\n      model: lognormal\n      median_nanos: 9000000\n      seed: 5\n"
    )
    cfg = SessionConfig.from_yaml(path)
    assert cfg.seed == 42
    assert cfg.profiles["default"].network_latency.seed == 5
    assert cfg.profiles["default"].network_latency.median_nanos == 9_000_000


def test_self_trade_prevention_written_and_parsed(tmp_path):
    profile = SimulationProfile(self_trade_prevention="CANCEL_RESTING")
    props = profile.to_properties("simulation.profiles.kalshi")
    assert props["simulation.profiles.kalshi.self.trade.prevention"] == "CANCEL_RESTING"

    path = tmp_path / "session.yaml"
    path.write_text(
        "mode: paper\n"
        "listings:\n  - listing_id: 7\n    profile: kalshi\n"
        "profiles:\n  kalshi:\n    self_trade_prevention: CANCEL_RESTING\n"
    )
    assert SessionConfig.from_yaml(path).profiles["kalshi"].self_trade_prevention == "CANCEL_RESTING"
