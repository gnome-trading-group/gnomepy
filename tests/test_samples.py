"""The samples reader against a fake samples API that pages as the real one does."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gnomepy import samples as samples_module
from gnomepy.samples import explode_samples, load_samples, load_session_samples


class _Response:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


@pytest.fixture
def api(monkeypatch):
    stored = [
        {"kind": "runtime", "tsMs": 1_000, "data": {"host": {"cpuBusyPermille": 120}, "threads": [
            {"name": "oms", "cycles": 10}, {"name": "strategy", "cycles": 5}]}},
        {"kind": "runtime", "tsMs": 11_000, "data": {"host": {"cpuBusyPermille": 140}, "threads": [
            {"name": "oms", "cycles": 12}]}},
        {"kind": "runtime", "tsMs": 21_000, "data": {"host": {"cpuBusyPermille": 90}, "threads": []}},
    ]
    calls = []

    def get(url, params, headers, timeout):
        calls.append(dict(params))
        after = params.get("after")
        start = 0 if after is None else int(after.split(":")[0])
        remaining = [s for s in stored if s["tsMs"] > start or after is None]
        page = remaining[:2]
        more = len(remaining) > 2
        return _Response({"samples": page, "next": f"{page[-1]['tsMs']}:runtime" if more else None})

    monkeypatch.setattr(samples_module.requests, "get", get)
    return calls


def test_pages_until_there_is_no_next_and_flattens_the_data(api):
    frame = load_samples("gateway", "live:kalshi:main", kind="runtime", api_key="k")

    assert list(frame["host.cpuBusyPermille"]) == [120, 140, 90]
    assert frame.index[0] == pd.Timestamp(1_000, unit="ms", tz="UTC")
    assert list(frame["kind"]) == ["runtime"] * 3
    assert len(api) == 2
    assert api[0]["subjectKind"] == "gateway" and api[0]["subjectId"] == "live:kalshi:main"
    assert api[1]["after"] == "11000:runtime"


def test_a_list_column_explodes_to_a_row_per_element(api):
    threads = explode_samples(load_session_samples("s1", kind="runtime", api_key="k"), "threads")

    assert list(threads["name"]) == ["oms", "strategy", "oms"]
    assert list(threads["cycles"]) == [10, 5, 12]


def test_times_are_sent_as_epoch_milliseconds(api):
    load_samples("shard", "use1-1b7a9c", kind_prefix="shard.", start="1970-01-01T00:00:01Z", end=5_000, api_key="k")

    assert api[0]["from"] == 1_000
    assert api[0]["to"] == 5_000
    assert api[0]["kindPrefix"] == "shard."


def test_exactly_one_of_kind_or_prefix():
    with pytest.raises(ValueError):
        load_samples("session", "s1", api_key="k")
    with pytest.raises(ValueError):
        load_samples("session", "s1", kind="runtime", kind_prefix="r", api_key="k")


def test_numpy_integers_and_digit_strings_are_milliseconds(api):
    load_samples("shard", "use1-1b7a9c", kind="runtime", start=np.int64(1_000), end="5000", api_key="k")

    assert api[0]["from"] == 1_000
    assert api[0]["to"] == 5_000


def test_a_list_of_kinds_is_sent_as_kinds(api):
    load_samples("session", "s1", kind=["runtime", "session.ledger"], api_key="k")

    assert api[0]["kinds"] == "runtime,session.ledger"
    assert "kind" not in api[0]
