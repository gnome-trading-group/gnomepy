"""Sweep syntax expansion for backtest YAML configs."""
from __future__ import annotations

import copy
import itertools
import json
from typing import Any


def _linspace(min_val: float, max_val: float, step: float) -> list[float]:
    values = []
    v = min_val
    while v <= max_val + step * 1e-9:
        values.append(round(v, 10))
        v += step
    return values


def _is_sweep_range(value: Any) -> bool:
    return isinstance(value, dict) and {"min", "max", "step"} <= value.keys()


def _collect_sweeps(args: dict) -> dict[str, list]:
    sweeps: dict[str, list] = {}
    for key, value in args.items():
        if isinstance(value, list):
            sweeps[key] = value
        elif _is_sweep_range(value):
            sweeps[key] = _linspace(value["min"], value["max"], value["step"])
    return sweeps


def _collect_sweeps_recursive(d: dict, prefix: str) -> dict[str, list]:
    sweeps: dict[str, list] = {}
    for key, value in d.items():
        full_key = f"{prefix}.{key}"
        if isinstance(value, list):
            sweeps[full_key] = value
        elif _is_sweep_range(value):
            sweeps[full_key] = _linspace(value["min"], value["max"], value["step"])
        elif isinstance(value, dict):
            sweeps.update(_collect_sweeps_recursive(value, full_key))
    return sweeps


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def get_param_value(config: dict, key: str) -> str:
    """Extract a sweep parameter value from an expanded config by key.

    Strategy arg keys are flat (e.g., ``ewma_alpha``); profile keys use
    dot-notation rooted at ``profiles`` (e.g., ``profiles.default.fee_model.taker_fee``).
    Dict values (e.g., a full sub-model config) are serialized as JSON.
    """
    if "." in key:
        try:
            keys = key.split(".")
            val: Any = config
            for k in keys:
                val = val[k]
        except (KeyError, TypeError):
            val = ""
    else:
        val = config.get("strategy", {}).get("args", {}).get(key, "")
    if isinstance(val, dict):
        return json.dumps(val, sort_keys=True)
    return str(val)


def expand_sweep(config: dict) -> list[dict]:
    """Expand sweep syntax in strategy.args and profiles into a list of individual configs.

    List values and {min,max,step} ranges in strategy.args or within any profile
    are expanded into the cartesian product. Scalar values are fixed across all jobs.
    Returns [config] if no sweep parameters are found.
    """
    strategy_args = config.get("strategy", {}).get("args", {})
    arg_sweeps = _collect_sweeps(strategy_args)

    profile_sweeps = _collect_sweeps_recursive(config.get("profiles", {}), "profiles")

    all_sweeps = {**arg_sweeps, **profile_sweeps}

    if not all_sweeps:
        return [copy.deepcopy(config)]

    keys = list(all_sweeps.keys())
    value_lists = [all_sweeps[k] for k in keys]

    expanded = []
    for combo in itertools.product(*value_lists):
        c = copy.deepcopy(config)
        for key, val in zip(keys, combo):
            if key in arg_sweeps:
                c["strategy"]["args"][key] = val
            else:
                _set_nested(c, key, val)
        expanded.append(c)

    return expanded


def sweep_params(config: dict) -> dict[str, list]:
    """Return the swept parameter names and their candidate values."""
    arg_sweeps = _collect_sweeps(config.get("strategy", {}).get("args", {}))
    profile_sweeps = _collect_sweeps_recursive(config.get("profiles", {}), "profiles")
    return {**arg_sweeps, **profile_sweeps}
