"""DatasetStore against a local SQLite catalog — no AWS required."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pyarrow as pa
import pytest
from pyiceberg.expressions import EqualTo

from gnomepy.data import CatalogConfig, DatasetStore, TableSpec

T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
T1 = T0 + timedelta(days=4)

SPEC = TableSpec(
    "player_stats",
    pa.schema([
        pa.field("match_id", pa.int64(), nullable=False),
        pa.field("player_id", pa.int64(), nullable=False),
        pa.field("match_date", pa.timestamp("us")),
        pa.field("kickoff", pa.timestamp("us", tz="UTC")),
        pa.field("rating", pa.float64()),
        pa.field("note", pa.string()),
    ]),
    key=("match_id", "player_id"),
    partition_by=("match_date", "month"),
)


def frame(rows: list[tuple[int, int, float]]) -> pd.DataFrame:
    return pd.DataFrame({
        "match_id": [r[0] for r in rows],
        "player_id": [r[1] for r in rows],
        "match_date": pd.to_datetime(["2026-09-30"] * len(rows)),
        "kickoff": pd.to_datetime(["2026-09-30T18:00:00.123456789Z"] * len(rows)),
        "rating": [r[2] for r in rows],
        "note": [None] * len(rows),
    })


@pytest.fixture
def store(tmp_path):
    return DatasetStore(CatalogConfig(local_warehouse=tmp_path), code_version="test@v1", run_id="run-1")


def test_publish_and_load_roundtrip_keeps_values_and_types(store):
    result = store.publish(SPEC, frame([(1, 10, 1.1), (1, 11, 0.9)]), ingested_at=T0)

    loaded = store.load("player_stats")

    assert result.rows_written == 2 and result.rows_skipped == 0
    assert loaded[["match_id", "player_id", "rating"]].values.tolist() == [[1, 10, 1.1], [1, 11, 0.9]]
    assert str(loaded.match_date.dtype) == "datetime64[us]"
    assert str(loaded.kickoff.dtype) == "datetime64[us, UTC]"
    assert loaded.note.isna().all()


def test_all_null_column_writes_because_schema_is_declared(store):
    store.publish(SPEC, frame([(1, 10, 1.0)]), ingested_at=T0)

    assert store.load("player_stats").note.isna().all()


def test_republishing_identical_rows_writes_nothing(store):
    store.publish(SPEC, frame([(1, 10, 1.1), (1, 11, 0.9)]), ingested_at=T0)

    result = store.publish(SPEC, frame([(1, 10, 1.1), (1, 11, 0.9)]), ingested_at=T1)

    assert result.rows_written == 0 and result.rows_skipped == 2


def test_correction_appends_and_as_of_returns_the_table_as_it_was_known(store):
    store.publish(SPEC, frame([(1, 10, 1.1), (1, 11, 0.9)]), ingested_at=T0)
    result = store.publish(SPEC, frame([(1, 10, 1.5), (1, 11, 0.9), (2, 10, 1.0)]), ingested_at=T1)

    before = store.load("player_stats", as_of=T0 + timedelta(days=1))
    after = store.load("player_stats")

    assert result.rows_written == 2
    assert before[["match_id", "player_id", "rating"]].values.tolist() == [[1, 10, 1.1], [1, 11, 0.9]]
    assert after[["match_id", "player_id", "rating"]].values.tolist() == [[1, 10, 1.5], [1, 11, 0.9], [2, 10, 1.0]]


def test_exact_duplicates_in_one_publish_are_collapsed(store):
    result = store.publish(SPEC, frame([(1, 10, 1.1), (1, 10, 1.1)]), ingested_at=T0)

    assert result.rows_written == 1
    assert len(store.load("player_stats")) == 1


def test_conflicting_values_for_one_key_in_one_publish_are_rejected(store):
    with pytest.raises(ValueError, match="more than once"):
        store.publish(SPEC, frame([(1, 10, 1.1), (1, 10, 2.0)]), ingested_at=T0)


def test_column_mismatch_is_rejected(store):
    with pytest.raises(ValueError, match="unexpected \\['extra'\\]"):
        store.publish(SPEC, frame([(1, 10, 1.1)]).assign(extra=1), ingested_at=T0)


def test_evolve_adds_a_column_that_old_rows_read_as_null(store):
    store.publish(SPEC, frame([(1, 10, 1.1)]), ingested_at=T0)
    wider = TableSpec(SPEC.name, SPEC.schema.append(pa.field("adr", pa.float64())), SPEC.key, SPEC.partition_by)

    store.evolve(wider)
    store.publish(wider, frame([(2, 10, 1.0)]).assign(adr=80.0), ingested_at=T1)
    loaded = store.load("player_stats")

    assert loaded.adr.isna().tolist() == [True, False]


def test_metadata_columns_record_ingestion_and_code_version(store):
    store.publish(SPEC, frame([(1, 10, 1.1)]), ingested_at=T0)

    loaded = store.load("player_stats", include_metadata=True)

    assert loaded.ingested_at.iloc[0] == pd.Timestamp(T0)
    assert loaded.code_version.iloc[0] == "test@v1"


def test_snapshot_properties_carry_run_and_code_version(store):
    store.publish(SPEC, frame([(1, 10, 1.1)]), ingested_at=T0)

    summary = store._catalog.load_table("research.player_stats").current_snapshot().summary

    assert summary["gnome.run_id"] == "run-1"
    assert summary["gnome.code_version"] == "test@v1"


def test_loads_are_recorded_with_a_stable_rows_hash(store):
    store.publish(SPEC, frame([(1, 10, 1.1), (1, 11, 0.9)]), ingested_at=T0)

    store.load("player_stats", as_of=T1)
    store.load("player_stats", as_of=T1, row_filter=EqualTo("match_id", 1))

    first, second = store.loaded
    assert first.dataset == "player_stats" and first.row_count == 2 and first.as_of == T1
    assert first.rows_hash == second.rows_hash


def test_column_selection_keeps_key(store):
    store.publish(SPEC, frame([(1, 10, 1.1)]), ingested_at=T0)

    loaded = store.load("player_stats", columns=["rating"])

    assert list(loaded.columns) == ["match_id", "player_id", "rating"]


def test_list_columns_roundtrip_and_dedupe(store):
    spec = TableSpec("lineups", pa.schema([pa.field("match_id", pa.int64(), nullable=False),
                                           pa.field("player_ids", pa.list_(pa.int64()))]), key=("match_id",))
    store.publish(spec, pd.DataFrame({"match_id": [1], "player_ids": [[7, 8, 9]]}), ingested_at=T0)

    loaded = store.load("lineups")
    again = store.publish(spec, loaded, ingested_at=T1)

    assert list(loaded.player_ids.iloc[0]) == [7, 8, 9]
    assert again.rows_written == 0


def test_spec_rejects_nanosecond_timestamps():
    with pytest.raises(ValueError, match="microsecond"):
        TableSpec("bad", pa.schema([pa.field("t", pa.timestamp("ns"))]), key=("t",))
