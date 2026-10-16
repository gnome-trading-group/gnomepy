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


def _collect_sweeps_from_section(sweep_section: dict) -> tuple[dict[str, list], dict[str, list]]:
    """Extract strategy arg sweeps and simulation sweeps from the top-level sweep section.

    sweep.strategy keys map to strategy.args overrides.
    sweep.exchanges mirrors the exchanges structure (``exchanges.KALSHI.queue_model.cancel_ahead_probability``);
    sweep.listings is keyed by listing id and sets that listing's overrides
    (``listings.48212.network_latency.latency_nanos``).
    """
    arg_sweeps: dict[str, list] = {}
    for key, value in sweep_section.get("strategy", {}).items():
        if isinstance(value, list):
            arg_sweeps[key] = value
        elif _is_sweep_range(value):
            arg_sweeps[key] = _linspace(value["min"], value["max"], value["step"])

    simulation_sweeps = {
        **_collect_sweeps_recursive(sweep_section.get("exchanges", {}), "exchanges"),
        **_collect_sweeps_recursive(sweep_section.get("listings", {}), "listings"),
    }
    return arg_sweeps, simulation_sweeps


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


def _listing_overrides(config: dict, listing_id: str) -> dict:
    for entry in config.get("listings", []):
        if str(entry.get("listing_id")) == listing_id:
            return entry.setdefault("overrides", {})
    raise KeyError(f"sweep names listing {listing_id}, which the config doesn't trade")


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    # A swept block may not be overridden yet: it is created, and must then name its model (`type`) like any other.
    if keys[0] == "listings":
        d, keys = _listing_overrides(d, keys[1]), keys[2:]
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def get_param_value(config: dict, key: str) -> str:
    """Extract a sweep parameter value from an expanded config by key.

    Strategy arg keys are flat (e.g., ``ewma_alpha``); simulation keys use dot-notation rooted at ``exchanges``
    (``exchanges.KALSHI.fee_model.taker_fee``) or ``listings`` (``listings.48212.network_latency.latency_nanos``).
    Dict values (e.g., a full sub-model config) are serialized as JSON.
    """
    if "." in key:
        try:
            keys = key.split(".")
            val: Any = config
            if keys[0] == "listings":
                val = next(e for e in config["listings"] if str(e.get("listing_id")) == keys[1])["overrides"]
                keys = keys[2:]
            for k in keys:
                val = val[k]
        except (KeyError, TypeError, StopIteration):
            val = ""
    else:
        val = config.get("strategy", {}).get("args", {}).get(key, "")
    if isinstance(val, dict):
        return json.dumps(val, sort_keys=True)
    return str(val)


def expand_sweep(config: dict) -> list[dict]:
    """Expand the top-level ``sweep`` section into a list of individual configs.

    ``sweep.strategy`` keys override ``strategy.args`` values per job.
    ``sweep.exchanges`` and ``sweep.listings`` keys override simulation leaf values per job (see
    ``_collect_sweeps_from_section``). Everything else is fixed and passed through as-is — only the ``sweep``
    section is expanded.

    Returns ``[config]`` (with ``sweep`` stripped) when no sweep parameters are found.
    """
    sweep_section = config.get("sweep", {})
    arg_sweeps, simulation_sweeps = _collect_sweeps_from_section(sweep_section)
    all_sweeps = {**arg_sweeps, **simulation_sweeps}

    if not all_sweeps:
        c = copy.deepcopy(config)
        c.pop("sweep", None)
        return [c]

    keys = list(all_sweeps.keys())
    value_lists = [all_sweeps[k] for k in keys]

    expanded = []
    for combo in itertools.product(*value_lists):
        c = copy.deepcopy(config)
        c.pop("sweep", None)
        for key, val in zip(keys, combo):
            if key in arg_sweeps:
                c["strategy"]["args"][key] = val
            else:
                _set_nested(c, key, val)
        expanded.append(c)

    return expanded


def sweep_params(config: dict) -> dict[str, list]:
    """Return the swept parameter names and their candidate values.

    When scenarios are present, collects sweeps from the first scenario's flattened
    config since sweep dimensions are shared across all scenarios.
    """
    if "scenarios" in config:
        scenario_configs = expand_scenarios(config)
        if not scenario_configs:
            return {}
        _, first = scenario_configs[0]
        sweep_section = first.get("sweep", {})
    else:
        sweep_section = config.get("sweep", {})

    arg_sweeps, simulation_sweeps = _collect_sweeps_from_section(sweep_section)
    return {**arg_sweeps, **simulation_sweeps}


def scenario_names(config: dict) -> list[str]:
    """Return the list of scenario names defined in the config, or [] if none."""
    return list(config.get("scenarios", {}).keys())


def expand_scenarios(config: dict) -> list[tuple[str, dict]]:
    """Expand the scenarios section into (name, flat_config) pairs.

    Each flat config has top-level listings/start_date/end_date from the scenario,
    with scenario strategy_args shallow-merged into strategy.args and each scenario
    exchange's blocks laid over the top-level ones for that exchange.

    Returns [("", config)] when no scenarios key is present, preserving backward
    compatibility with single-scenario configs.
    """
    if "scenarios" not in config:
        return [("", copy.deepcopy(config))]

    result = []
    for name, scenario in config["scenarios"].items():
        c = {k: copy.deepcopy(v) for k, v in config.items() if k != "scenarios"}

        c["start_date"] = scenario["start_date"]
        c["end_date"] = scenario["end_date"]
        c["listings"] = copy.deepcopy(scenario["listings"])

        if "strategy_args" in scenario:
            if "strategy" not in c:
                c["strategy"] = {"args": {}}
            if "args" not in c["strategy"]:
                c["strategy"]["args"] = {}
            c["strategy"]["args"] = {**c["strategy"]["args"], **copy.deepcopy(scenario["strategy_args"])}

        if "exchanges" in scenario:
            exchanges = c.get("exchanges", {})
            for code, blocks in scenario["exchanges"].items():
                exchanges[code] = {**exchanges.get(code, {}), **copy.deepcopy(blocks)}
            c["exchanges"] = exchanges

        result.append((name, c))

    return result


def expand_scenarios_and_sweeps(config: dict) -> list[tuple[str, dict]]:
    """Expand scenarios × sweep params into (scenario_name, flat_config) pairs.

    Total jobs = len(scenarios) × len(sweep_combinations). Jobs are ordered
    scenario-first: all sweep combinations for scenario 0, then scenario 1, etc.

    Falls back to expand_sweep behavior when no scenarios are present (scenario_name
    will be "" for all jobs).
    """
    result = []
    for scenario_name, scenario_config in expand_scenarios(config):
        for expanded in expand_sweep(scenario_config):
            result.append((scenario_name, expanded))
    return result
