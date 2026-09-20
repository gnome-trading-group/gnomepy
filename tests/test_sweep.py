"""Tests for sweep.py — no JVM required."""
from __future__ import annotations

import json

import pytest

from gnomepy.sweep import (
    _collect_sweeps_recursive,
    _set_nested,
    expand_scenarios,
    expand_scenarios_and_sweeps,
    expand_sweep,
    get_param_value,
    scenario_names,
    sweep_params,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_config(**strategy_args):
    return {
        "strategy": {"class_name": "test:Strategy", "args": strategy_args},
        "profiles": {
            "default": {
                "fee_model": {"type": "static", "taker_fee": 0.0005, "maker_fee": -0.0002},
                "network_latency": {"type": "static", "latency_nanos": 5_000_000},
                "order_processing_latency": {"type": "static", "latency_nanos": 1_000_000},
                "queue_model": {"type": "risk_averse"},
            }
        },
    }


def _with_sweep(config: dict, **sweep_strategy_args) -> dict:
    """Add a sweep.strategy section to a config."""
    config = dict(config)
    config["sweep"] = {"strategy": sweep_strategy_args}
    return config


def _with_profile_sweep(config: dict, profiles_sweep: dict) -> dict:
    """Add a sweep.profiles section to a config."""
    config = dict(config)
    config.setdefault("sweep", {})["profiles"] = profiles_sweep
    return config


# ---------------------------------------------------------------------------
# _set_nested
# ---------------------------------------------------------------------------

def test_set_nested_single_level():
    d = {"a": 1}
    _set_nested(d, "a", 99)
    assert d["a"] == 99


def test_set_nested_multi_level():
    d = {"profiles": {"default": {"fee_model": {"taker_fee": 0.0005}}}}
    _set_nested(d, "profiles.default.fee_model.taker_fee", 0.001)
    assert d["profiles"]["default"]["fee_model"]["taker_fee"] == 0.001


# ---------------------------------------------------------------------------
# _collect_sweeps_recursive
# ---------------------------------------------------------------------------

def test_collect_recursive_leaf_list():
    profiles = {"default": {"network_latency": {"type": "static", "latency_nanos": [0, 5_000_000]}}}
    result = _collect_sweeps_recursive(profiles, "profiles")
    assert result == {"profiles.default.network_latency.latency_nanos": [0, 5_000_000]}


def test_collect_recursive_leaf_range():
    profiles = {"default": {"network_latency": {"type": "static", "latency_nanos": {"min": 0, "max": 10_000_000, "step": 5_000_000}}}}
    result = _collect_sweeps_recursive(profiles, "profiles")
    assert result == {"profiles.default.network_latency.latency_nanos": [0.0, 5_000_000.0, 10_000_000.0]}


def test_collect_recursive_list_of_dicts():
    profiles = {
        "default": {
            "network_latency": [
                {"type": "static", "latency_nanos": 5_000_000},
                {"type": "gaussian", "mu": 5_000_000.0, "sigma": 1_000_000.0},
            ]
        }
    }
    result = _collect_sweeps_recursive(profiles, "profiles")
    assert "profiles.default.network_latency" in result
    assert len(result["profiles.default.network_latency"]) == 2


def test_collect_recursive_no_sweeps():
    profiles = {"default": {"network_latency": {"type": "static", "latency_nanos": 5_000_000}}}
    result = _collect_sweeps_recursive(profiles, "profiles")
    assert result == {}


def test_collect_recursive_multiple_profiles():
    profiles = {
        "a": {"network_latency": {"type": "static", "latency_nanos": [1, 2]}},
        "b": {"fee_model": {"type": "static", "taker_fee": [0.001, 0.002]}},
    }
    result = _collect_sweeps_recursive(profiles, "profiles")
    assert "profiles.a.network_latency.latency_nanos" in result
    assert "profiles.b.fee_model.taker_fee" in result


# ---------------------------------------------------------------------------
# expand_sweep — strategy arg sweeps via sweep section
# ---------------------------------------------------------------------------

def test_expand_no_sweep():
    config = _base_config(alpha=0.9)
    result = expand_sweep(config)
    assert len(result) == 1
    assert result[0]["strategy"]["args"]["alpha"] == 0.9


def test_expand_strategy_list():
    config = _base_config(alpha=0.9)
    config["sweep"] = {"strategy": {"alpha": [0.9, 0.95, 0.99]}}
    result = expand_sweep(config)
    assert len(result) == 3
    alphas = [r["strategy"]["args"]["alpha"] for r in result]
    assert alphas == [0.9, 0.95, 0.99]


def test_expand_strategy_range():
    config = _base_config(threshold=1.0)
    config["sweep"] = {"strategy": {"threshold": {"min": 1.0, "max": 3.0, "step": 1.0}}}
    result = expand_sweep(config)
    assert len(result) == 3
    thresholds = [r["strategy"]["args"]["threshold"] for r in result]
    assert thresholds == [1.0, 2.0, 3.0]


def test_expand_strategy_cartesian():
    config = _base_config(alpha=0.9, threshold=1.0)
    config["sweep"] = {"strategy": {"alpha": [0.9, 0.95], "threshold": [1.0, 2.0]}}
    result = expand_sweep(config)
    assert len(result) == 4


def test_sweep_overrides_args_default():
    config = _base_config(alpha=0.9)
    config["sweep"] = {"strategy": {"alpha": [0.95, 0.99]}}
    result = expand_sweep(config)
    alphas = [r["strategy"]["args"]["alpha"] for r in result]
    assert alphas == [0.95, 0.99]


def test_expanded_config_has_no_sweep_key():
    config = _base_config(alpha=0.9)
    config["sweep"] = {"strategy": {"alpha": [0.9, 0.95]}}
    for r in expand_sweep(config):
        assert "sweep" not in r


def test_list_in_args_not_swept():
    """Lists in strategy.args are passed through as-is — not expanded."""
    config = _base_config(outcomes=[{"pm": 1}, {"pm": 2}], taker_labels=["pm"])
    result = expand_sweep(config)
    assert len(result) == 1
    assert result[0]["strategy"]["args"]["outcomes"] == [{"pm": 1}, {"pm": 2}]
    assert result[0]["strategy"]["args"]["taker_labels"] == ["pm"]


def test_no_sweep_section_produces_single_run():
    """Config with no sweep section always produces exactly one run."""
    config = _base_config(alpha=[0.9, 0.95], threshold={"min": 1, "max": 3, "step": 1})
    result = expand_sweep(config)
    assert len(result) == 1
    assert result[0]["strategy"]["args"]["alpha"] == [0.9, 0.95]


# ---------------------------------------------------------------------------
# expand_sweep — profile sweeps via sweep section
# ---------------------------------------------------------------------------

def test_expand_profile_leaf_sweep():
    config = _base_config()
    config["sweep"] = {"profiles": {"default": {"network_latency": {"latency_nanos": [0, 5_000_000]}}}}
    result = expand_sweep(config)
    assert len(result) == 2
    latencies = [r["profiles"]["default"]["network_latency"]["latency_nanos"] for r in result]
    assert latencies == [0, 5_000_000]


def test_expand_profile_range_sweep():
    config = _base_config()
    config["sweep"] = {"profiles": {"default": {"fee_model": {"taker_fee": {"min": 0.0003, "max": 0.0005, "step": 0.0001}}}}}
    result = expand_sweep(config)
    assert len(result) == 3
    fees = [r["profiles"]["default"]["fee_model"]["taker_fee"] for r in result]
    assert fees == [0.0003, 0.0004, 0.0005]


def test_expand_profile_list_of_dicts():
    config = _base_config()
    config["sweep"] = {
        "profiles": {
            "default": {
                "network_latency": [
                    {"type": "static", "latency_nanos": 5_000_000},
                    {"type": "gaussian", "mu": 5_000_000.0, "sigma": 1_000_000.0},
                ]
            }
        }
    }
    result = expand_sweep(config)
    assert len(result) == 2
    assert result[0]["profiles"]["default"]["network_latency"] == {"type": "static", "latency_nanos": 5_000_000}
    assert result[1]["profiles"]["default"]["network_latency"]["type"] == "gaussian"


def test_expand_list_of_dicts_does_not_share_references():
    config = _base_config()
    config["sweep"] = {
        "profiles": {
            "default": {
                "network_latency": [
                    {"type": "static", "latency_nanos": 5_000_000},
                    {"type": "static", "latency_nanos": 10_000_000},
                ]
            }
        }
    }
    result = expand_sweep(config)
    result[0]["profiles"]["default"]["network_latency"]["latency_nanos"] = 999
    assert result[1]["profiles"]["default"]["network_latency"]["latency_nanos"] == 10_000_000


def test_expand_combined_strategy_and_profile():
    config = _base_config(alpha=0.9)
    config["sweep"] = {
        "strategy": {"alpha": [0.9, 0.95]},
        "profiles": {"default": {"network_latency": {"latency_nanos": [0, 5_000_000]}}},
    }
    result = expand_sweep(config)
    assert len(result) == 4
    combos = [
        (r["strategy"]["args"]["alpha"], r["profiles"]["default"]["network_latency"]["latency_nanos"])
        for r in result
    ]
    assert (0.9, 0) in combos
    assert (0.9, 5_000_000) in combos
    assert (0.95, 0) in combos
    assert (0.95, 5_000_000) in combos


def test_expand_multiple_profiles():
    config = {
        "strategy": {"class_name": "test:Arb", "args": {}},
        "profiles": {
            "exchange_a": {"network_latency": {"type": "static", "latency_nanos": 1_000_000}},
            "exchange_b": {"network_latency": {"type": "static", "latency_nanos": 2_000_000}},
        },
        "sweep": {
            "profiles": {
                "exchange_a": {"network_latency": {"latency_nanos": [1_000_000, 5_000_000]}},
                "exchange_b": {"network_latency": {"latency_nanos": [2_000_000, 8_000_000]}},
            }
        },
    }
    result = expand_sweep(config)
    assert len(result) == 4


def test_expand_preserves_non_sweep_profile_fields():
    config = _base_config()
    config["sweep"] = {"profiles": {"default": {"network_latency": {"latency_nanos": [0, 5_000_000]}}}}
    result = expand_sweep(config)
    for r in result:
        assert r["profiles"]["default"]["fee_model"]["taker_fee"] == 0.0005
        assert r["profiles"]["default"]["network_latency"]["type"] == "static"


# ---------------------------------------------------------------------------
# sweep_params
# ---------------------------------------------------------------------------

def test_sweep_params_strategy_only():
    config = _base_config(alpha=0.9)
    config["sweep"] = {"strategy": {"alpha": [0.9, 0.95]}}
    params = sweep_params(config)
    assert params == {"alpha": [0.9, 0.95]}


def test_sweep_params_profile_only():
    config = _base_config()
    config["sweep"] = {"profiles": {"default": {"network_latency": {"latency_nanos": [0, 5_000_000]}}}}
    params = sweep_params(config)
    assert params == {"profiles.default.network_latency.latency_nanos": [0, 5_000_000]}


def test_sweep_params_combined():
    config = _base_config(alpha=0.9)
    config["sweep"] = {
        "strategy": {"alpha": [0.9, 0.95]},
        "profiles": {"default": {"network_latency": {"latency_nanos": [0, 5_000_000]}}},
    }
    params = sweep_params(config)
    assert "alpha" in params
    assert "profiles.default.network_latency.latency_nanos" in params


def test_sweep_params_empty():
    config = _base_config(alpha=0.9)
    assert sweep_params(config) == {}


# ---------------------------------------------------------------------------
# get_param_value
# ---------------------------------------------------------------------------

def test_get_param_value_flat_key():
    config = _base_config(alpha=0.95)
    assert get_param_value(config, "alpha") == "0.95"


def test_get_param_value_dotted_key():
    config = _base_config()
    assert get_param_value(config, "profiles.default.network_latency.latency_nanos") == "5000000"


def test_get_param_value_dict_value():
    config = _base_config()
    val = get_param_value(config, "profiles.default.network_latency")
    parsed = json.loads(val)
    assert parsed["type"] == "static"
    assert parsed["latency_nanos"] == 5_000_000


def test_get_param_value_missing_key():
    config = _base_config()
    assert get_param_value(config, "profiles.default.nonexistent.field") == ""


# ---------------------------------------------------------------------------
# scenario helpers
# ---------------------------------------------------------------------------

def _scenario_config(**strategy_args):
    """Base config with two scenarios but no sweep params."""
    return {
        "strategy": {"class_name": "test:Arb", "args": {"max_position": 100, **strategy_args}},
        "profiles": {
            "exchange_a": {
                "fee_model": {"type": "static", "taker_fee": 0.07, "maker_fee": 0.0},
                "network_latency": {"type": "static", "latency_nanos": 50_000_000},
                "order_processing_latency": {"type": "static", "latency_nanos": 5_000_000},
                "queue_model": {"type": "risk_averse"},
            }
        },
        "scenarios": {
            "baseball": {
                "start_date": "2026-08-24T00:00:00",
                "end_date": "2026-08-24T02:00:00",
                "listings": [
                    {"listing_id": 100, "profile": "exchange_a"},
                    {"listing_id": 101, "profile": "exchange_a"},
                ],
                "strategy_args": {"event_ids": [1, 2]},
            },
            "football": {
                "start_date": "2026-09-01T00:00:00",
                "end_date": "2026-09-01T03:00:00",
                "listings": [
                    {"listing_id": 200, "profile": "exchange_a"},
                    {"listing_id": 201, "profile": "exchange_a"},
                ],
                "strategy_args": {"event_ids": [3, 4]},
            },
        },
    }


# ---------------------------------------------------------------------------
# scenario_names
# ---------------------------------------------------------------------------

def test_scenario_names_with_scenarios():
    config = _scenario_config()
    assert scenario_names(config) == ["baseball", "football"]


def test_scenario_names_without_scenarios():
    config = _base_config(alpha=0.9)
    assert scenario_names(config) == []


def test_scenario_names_empty_scenarios():
    config = {"strategy": {"class_name": "test:S", "args": {}}, "scenarios": {}}
    assert scenario_names(config) == []


# ---------------------------------------------------------------------------
# expand_scenarios
# ---------------------------------------------------------------------------

def test_expand_scenarios_no_scenarios():
    config = _base_config(alpha=0.9)
    result = expand_scenarios(config)
    assert len(result) == 1
    name, cfg = result[0]
    assert name == ""
    assert cfg["strategy"]["args"]["alpha"] == 0.9
    assert "scenarios" not in cfg


def test_expand_scenarios_single():
    config = {
        "strategy": {"class_name": "test:Arb", "args": {"max_position": 100}},
        "profiles": {"exchange_a": {}},
        "scenarios": {
            "baseball": {
                "start_date": "2026-08-24T00:00:00",
                "end_date": "2026-08-24T02:00:00",
                "listings": [{"listing_id": 100, "profile": "exchange_a"}],
            }
        },
    }
    result = expand_scenarios(config)
    assert len(result) == 1
    name, cfg = result[0]
    assert name == "baseball"
    assert cfg["start_date"] == "2026-08-24T00:00:00"
    assert cfg["end_date"] == "2026-08-24T02:00:00"
    assert cfg["listings"] == [{"listing_id": 100, "profile": "exchange_a"}]
    assert "scenarios" not in cfg


def test_expand_scenarios_multiple():
    config = _scenario_config()
    result = expand_scenarios(config)
    assert len(result) == 2
    names = [name for name, _ in result]
    assert names == ["baseball", "football"]


def test_expand_scenarios_strategy_args_merge():
    config = _scenario_config()
    result = expand_scenarios(config)
    baseball_cfg = dict(result)["baseball"]
    assert baseball_cfg["strategy"]["args"]["event_ids"] == [1, 2]
    assert baseball_cfg["strategy"]["args"]["max_position"] == 100


def test_expand_scenarios_strategy_args_override():
    config = {
        "strategy": {"class_name": "test:Arb", "args": {"max_position": 100, "threshold": 5}},
        "profiles": {},
        "scenarios": {
            "test": {
                "start_date": "2026-01-01T00:00:00",
                "end_date": "2026-01-02T00:00:00",
                "listings": [],
                "strategy_args": {"threshold": 10},
            }
        },
    }
    _, cfg = expand_scenarios(config)[0]
    assert cfg["strategy"]["args"]["threshold"] == 10
    assert cfg["strategy"]["args"]["max_position"] == 100


def test_expand_scenarios_no_strategy_args():
    config = {
        "strategy": {"class_name": "test:Arb", "args": {"max_position": 100}},
        "profiles": {},
        "scenarios": {
            "test": {
                "start_date": "2026-01-01T00:00:00",
                "end_date": "2026-01-02T00:00:00",
                "listings": [],
            }
        },
    }
    _, cfg = expand_scenarios(config)[0]
    assert cfg["strategy"]["args"] == {"max_position": 100}


def test_expand_scenarios_profiles_merge():
    config = {
        "strategy": {"class_name": "test:Arb", "args": {}},
        "profiles": {"exchange_a": {"fee_model": {"type": "static", "taker_fee": 0.07}}},
        "scenarios": {
            "test": {
                "start_date": "2026-01-01T00:00:00",
                "end_date": "2026-01-02T00:00:00",
                "listings": [],
                "profiles": {"exchange_b": {"fee_model": {"type": "static", "taker_fee": 0.05}}},
            }
        },
    }
    _, cfg = expand_scenarios(config)[0]
    assert "exchange_a" in cfg["profiles"]
    assert "exchange_b" in cfg["profiles"]


def test_expand_scenarios_no_scenarios_key_in_output():
    config = _scenario_config()
    for _, cfg in expand_scenarios(config):
        assert "scenarios" not in cfg


def test_expand_scenarios_deep_copy_isolation():
    config = _scenario_config()
    result = expand_scenarios(config)
    _, cfg0 = result[0]
    cfg0["strategy"]["args"]["max_position"] = 999
    _, cfg1 = result[1]
    assert cfg1["strategy"]["args"]["max_position"] == 100


def test_expand_scenarios_preserves_top_level_fields():
    config = {**_scenario_config(), "record_depth": 10}
    for _, cfg in expand_scenarios(config):
        assert cfg["record_depth"] == 10
        assert "scenarios" not in cfg


# ---------------------------------------------------------------------------
# expand_scenarios_and_sweeps
# ---------------------------------------------------------------------------

def test_expand_scenarios_and_sweeps_scenarios_only():
    config = _scenario_config()
    result = expand_scenarios_and_sweeps(config)
    assert len(result) == 2
    names = [name for name, _ in result]
    assert names == ["baseball", "football"]


def test_expand_scenarios_and_sweeps_sweeps_only():
    config = _base_config(alpha=0.9)
    config["sweep"] = {"strategy": {"alpha": [0.9, 0.95]}}
    result = expand_scenarios_and_sweeps(config)
    assert len(result) == 2
    names = [name for name, _ in result]
    assert all(name == "" for name in names)


def test_expand_scenarios_and_sweeps_cartesian():
    config = _scenario_config()
    config["sweep"] = {"strategy": {"threshold": [5, 10, 15]}}
    result = expand_scenarios_and_sweeps(config)
    assert len(result) == 6  # 2 scenarios × 3 threshold values
    assert all(name == "baseball" for name, _ in result[:3])
    assert all(name == "football" for name, _ in result[3:])
    thresholds_baseball = [cfg["strategy"]["args"]["threshold"] for _, cfg in result[:3]]
    assert thresholds_baseball == [5, 10, 15]


def test_expand_scenarios_and_sweeps_no_scenarios_no_sweeps():
    config = _base_config(alpha=0.9)
    result = expand_scenarios_and_sweeps(config)
    assert len(result) == 1
    name, cfg = result[0]
    assert name == ""
    assert cfg["strategy"]["args"]["alpha"] == 0.9


def test_expand_scenarios_and_sweeps_scenario_has_no_scenarios_key():
    config = _scenario_config()
    config["sweep"] = {"strategy": {"threshold": [5, 10]}}
    for _, cfg in expand_scenarios_and_sweeps(config):
        assert "scenarios" not in cfg


# ---------------------------------------------------------------------------
# sweep_params with scenarios
# ---------------------------------------------------------------------------

def test_sweep_params_with_scenarios():
    config = _scenario_config()
    config["sweep"] = {"strategy": {"threshold": [5, 10, 15]}}
    params = sweep_params(config)
    assert "threshold" in params
    assert params["threshold"] == [5, 10, 15]


def test_sweep_params_with_scenarios_no_sweeps():
    config = {
        "strategy": {"class_name": "test:Arb", "args": {"max_position": 100}},
        "profiles": {},
        "scenarios": {
            "a": {"start_date": "2026-01-01T00:00:00", "end_date": "2026-01-02T00:00:00", "listings": []},
            "b": {"start_date": "2026-02-01T00:00:00", "end_date": "2026-02-02T00:00:00", "listings": []},
        },
    }
    params = sweep_params(config)
    assert params == {}
