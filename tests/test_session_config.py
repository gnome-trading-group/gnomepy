from __future__ import annotations

import json
from datetime import datetime

import pytest

from gnomepy.java.strategy.config import ListingSessionConfig, SessionConfig, StrategyConfig


def _paper(**kwargs) -> SessionConfig:
    return SessionConfig(mode="paper", listings=[ListingSessionConfig(listing_id=7)], **kwargs)


def test_a_session_with_no_simulation_settings_simulates_as_its_exchanges_are_recorded():
    assert _paper().to_session_config()["simulation"] == {}


def test_the_session_config_carries_overrides_in_the_backtest_shape(tmp_path):
    path = tmp_path / "session.yaml"
    path.write_text(
        "mode: paper\n"
        "seed: 42\n"
        "exchanges:\n  KALSHI:\n    queue_model: {type: optimistic}\n"
        "listings:\n"
        "  - listing_id: 7\n"
        "  - listing_id: 8\n    overrides:\n      network_latency: {type: static, latency_nanos: 1}\n"
    )
    config = SessionConfig.from_yaml(path).to_session_config()

    assert config["listings"] == [7, 8]
    assert config["simulation"] == {
        "seed": 42,
        "exchanges": {"KALSHI": {"queue_model": {"type": "optimistic"}}},
        "listings": {"8": {"network_latency": {"type": "static", "latency_nanos": 1}}},
    }


def test_a_deployed_session_is_pinned_by_the_registry_and_a_local_run_when_it_starts():
    assert "as_of" not in _paper().to_session_config()["simulation"]
    simulation = json.loads(_paper(seed=3).to_properties()["simulation.json"])
    assert datetime.fromisoformat(simulation["as_of"])
    assert simulation["seed"] == 3
    assert json.loads(_paper(simulation_as_of="2026-10-01T00:00:00Z").to_properties()["simulation.json"])["as_of"] == (
        "2026-10-01T00:00:00Z"
    )


def test_a_live_session_has_no_simulation():
    live = SessionConfig(mode="live", listings=[ListingSessionConfig(listing_id=7)])
    assert "simulation" not in live.to_session_config()
    assert "simulation.json" not in live.to_properties()


def test_a_python_class_is_sent_to_the_registry_but_handed_to_a_local_jvm_directly():
    config = _paper(strategy=StrategyConfig(class_name="mod.path:Cls", args={"edge": 0.1}))
    assert config.to_session_config()["strategy.class"] == "mod.path:Cls"
    assert "strategy.class" not in config.to_properties()
    assert config.to_properties()["strategy.args.edge"] == 0.1


def test_an_unknown_block_is_refused(tmp_path):
    path = tmp_path / "session.yaml"
    path.write_text("mode: paper\nlistings:\n  - listing_id: 7\nexchanges:\n  KALSHI:\n    fee: {model: static}\n")
    with pytest.raises(ValueError, match="exchanges.KALSHI has unknown blocks: fee"):
        SessionConfig.from_yaml(path)
