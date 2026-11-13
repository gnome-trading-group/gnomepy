"""Readings over time from the platform's processes: sessions, order gateways and collection shards.

Each process sends samples, each a kind (``runtime``, ``latency.oms``, ``session.positions``, ...), a moment and its
data, to the registry, which keeps seven days of them (gnome-registry ``ops.sample``). Read them through the samples
API, by subject: a session by its id, a gateway by its key (``live:kalshi:main``), a shard by its id
(``use1-1b7a9c``). Durations are nanoseconds and counters the change since the subject's previous sample of that kind.
"""

from __future__ import annotations

import numbers
from typing import Iterator, Sequence

import pandas as pd
import requests

from gnomepy.config import config, resolve_registry_api_key

REQUEST_TIMEOUT_SECONDS = 60
PAGE_LIMIT = 2_000


def iter_samples(
    subject_kind: str,
    subject_id: str,
    kind: str | Sequence[str] | None = None,
    kind_prefix: str | None = None,
    start: str | int | pd.Timestamp | None = None,
    end: str | int | pd.Timestamp | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Iterator[dict]:
    """Each sample of a subject, oldest first, as ``{"kind", "tsMs", "data"}``, paging through the API.

    Give exactly one of ``kind`` (one kind, or a list of them) or ``kind_prefix`` (``"latency."`` for every thread's
    latency). ``start`` is
    inclusive and ``end`` exclusive; either may be epoch milliseconds, an ISO time or a Timestamp.
    """
    if (kind is None) == (kind_prefix is None):
        raise ValueError("Give exactly one of kind or kind_prefix")
    url = (base_url or f"https://{config.SAMPLES_API_HOST}/api").rstrip("/") + "/samples"
    headers = {"x-api-key": api_key or resolve_registry_api_key()}
    params: dict[str, str | int] = {"subjectKind": subject_kind, "subjectId": subject_id, "limit": PAGE_LIMIT}
    if isinstance(kind, str):
        params["kind"] = kind
    elif kind is not None:
        params["kinds"] = ",".join(kind)
    else:
        params["kindPrefix"] = kind_prefix
    if start is not None:
        params["from"] = _epoch_ms(start)
    if end is not None:
        params["to"] = _epoch_ms(end)
    while True:
        response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        page = response.json()
        yield from page["samples"]
        if not page.get("next"):
            return
        params["after"] = page["next"]


def load_samples(
    subject_kind: str,
    subject_id: str,
    kind: str | Sequence[str] | None = None,
    kind_prefix: str | None = None,
    start: str | int | pd.Timestamp | None = None,
    end: str | int | pd.Timestamp | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> pd.DataFrame:
    """A subject's samples as a DataFrame indexed by UTC time, one row per sample.

    Nested data is flattened to dotted columns (``host.cpuBusyPermille``); lists (a ``runtime`` sample's ``threads``,
    a latency sample's ``keys``) stay as lists in their column, for :func:`explode_samples`. A ``kind`` column says
    which kind each row is when reading by prefix.
    """
    rows = list(iter_samples(subject_kind, subject_id, kind, kind_prefix, start, end, api_key, base_url))
    if not rows:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="ts"))
    frame = pd.json_normalize([row["data"] for row in rows])
    frame.insert(0, "kind", [row["kind"] for row in rows])
    frame.index = pd.DatetimeIndex(pd.to_datetime([row["tsMs"] for row in rows], unit="ms", utc=True), name="ts")
    return frame


def load_session_samples(session_id: str, kind: str | None = None, kind_prefix: str | None = None, **kwargs) -> pd.DataFrame:
    """A session's samples (see :func:`load_samples`)."""
    return load_samples("session", session_id, kind, kind_prefix, **kwargs)


def explode_samples(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """One row per element of a list column (a latency sample's ``keys``, a runtime sample's ``threads``), its fields
    flattened beside the sample's time and kind."""
    exploded = frame[["kind", column]].explode(column).dropna(subset=[column])
    if exploded.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="ts"))
    fields = pd.json_normalize(exploded[column].tolist())
    fields.index = exploded.index
    return pd.concat([exploded[["kind"]], fields], axis=1)


def _epoch_ms(value: str | int | pd.Timestamp) -> int:
    # Integers (numpy's too) and digit strings are epoch milliseconds, as the API takes them; anything else is a time.
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return int(timestamp.value // 1_000_000)
