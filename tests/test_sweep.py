"""Tests for sweep.py — no JVM required."""
from __future__ import annotations

import json

import pytest

from gnomepy.sweep import (
    _collect_sweeps_recursive,
    _set_nested,
    expand_sweep,
    get_param_value,
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
# expand_sweep — backward compat (strategy args only)
# ---------------------------------------------------------------------------

def test_expand_no_sweep():
    config = _base_config(alpha=0.9)
    result = expand_sweep(config)
    assert len(result) == 1
    assert result[0]["strategy"]["args"]["alpha"] == 0.9


def test_expand_strategy_list():
    config = _base_config(alpha=[0.9, 0.95, 0.99])
    result = expand_sweep(config)
    assert len(result) == 3
    alphas = [r["strategy"]["args"]["alpha"] for r in result]
    assert alphas == [0.9, 0.95, 0.99]


def test_expand_strategy_range():
    config = _base_config(threshold={"min": 1.0, "max": 3.0, "step": 1.0})
    result = expand_sweep(config)
    assert len(result) == 3
    thresholds = [r["strategy"]["args"]["threshold"] for r in result]
    assert thresholds == [1.0, 2.0, 3.0]


def test_expand_strategy_cartesian():
    config = _base_config(alpha=[0.9, 0.95], threshold=[1.0, 2.0])
    result = expand_sweep(config)
    assert len(result) == 4


# ---------------------------------------------------------------------------
# expand_sweep — profile sweeps
# ---------------------------------------------------------------------------

def test_expand_profile_leaf_sweep():
    config = _base_config()
    config["profiles"]["default"]["network_latency"]["latency_nanos"] = [0, 5_000_000]
    result = expand_sweep(config)
    assert len(result) == 2
    latencies = [r["profiles"]["default"]["network_latency"]["latency_nanos"] for r in result]
    assert latencies == [0, 5_000_000]


def test_expand_profile_range_sweep():
    config = _base_config()
    config["profiles"]["default"]["fee_model"]["taker_fee"] = {"min": 0.0003, "max": 0.0005, "step": 0.0001}
    result = expand_sweep(config)
    assert len(result) == 3
    fees = [r["profiles"]["default"]["fee_model"]["taker_fee"] for r in result]
    assert fees == [0.0003, 0.0004, 0.0005]


def test_expand_profile_list_of_dicts():
    config = _base_config()
    config["profiles"]["default"]["network_latency"] = [
        {"type": "static", "latency_nanos": 5_000_000},
        {"type": "gaussian", "mu": 5_000_000.0, "sigma": 1_000_000.0},
    ]
    result = expand_sweep(config)
    assert len(result) == 2
    assert result[0]["profiles"]["default"]["network_latency"] == {"type": "static", "latency_nanos": 5_000_000}
    assert result[1]["profiles"]["default"]["network_latency"]["type"] == "gaussian"


def test_expand_list_of_dicts_does_not_share_references():
    config = _base_config()
    config["profiles"]["default"]["network_latency"] = [
        {"type": "static", "latency_nanos": 5_000_000},
        {"type": "static", "latency_nanos": 10_000_000},
    ]
    result = expand_sweep(config)
    result[0]["profiles"]["default"]["network_latency"]["latency_nanos"] = 999
    assert result[1]["profiles"]["default"]["network_latency"]["latency_nanos"] == 10_000_000


def test_expand_combined_strategy_and_profile():
    config = _base_config(alpha=[0.9, 0.95])
    config["profiles"]["default"]["network_latency"]["latency_nanos"] = [0, 5_000_000]
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
            "exchange_a": {"network_latency": {"type": "static", "latency_nanos": [1_000_000, 5_000_000]}},
            "exchange_b": {"network_latency": {"type": "static", "latency_nanos": [2_000_000, 8_000_000]}},
        },
    }
    result = expand_sweep(config)
    assert len(result) == 4


def test_expand_preserves_non_sweep_profile_fields():
    config = _base_config()
    config["profiles"]["default"]["network_latency"]["latency_nanos"] = [0, 5_000_000]
    result = expand_sweep(config)
    for r in result:
        assert r["profiles"]["default"]["fee_model"]["taker_fee"] == 0.0005
        assert r["profiles"]["default"]["network_latency"]["type"] == "static"


# ---------------------------------------------------------------------------
# sweep_params
# ---------------------------------------------------------------------------

def test_sweep_params_strategy_only():
    config = _base_config(alpha=[0.9, 0.95])
    params = sweep_params(config)
    assert params == {"alpha": [0.9, 0.95]}


def test_sweep_params_profile_only():
    config = _base_config()
    config["profiles"]["default"]["network_latency"]["latency_nanos"] = [0, 5_000_000]
    params = sweep_params(config)
    assert params == {"profiles.default.network_latency.latency_nanos": [0, 5_000_000]}


def test_sweep_params_combined():
    config = _base_config(alpha=[0.9, 0.95])
    config["profiles"]["default"]["network_latency"]["latency_nanos"] = [0, 5_000_000]
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
