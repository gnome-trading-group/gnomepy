"""ArtifactStore against a local SQLite catalog and a local artifact root — no AWS required."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa
import pytest

from gnomepy.data import ArtifactQuery, ArtifactStore, CatalogConfig, DatasetStore, TableSpec

SPEC = TableSpec("priors", pa.schema([pa.field("match_id", pa.int64(), nullable=False), pa.field("x", pa.float64())]),
                 key=("match_id",))


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return CatalogConfig(local_warehouse=tmp_path / "warehouse", artifact_root=str(tmp_path / "files"))


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "model.joblib"
    path.write_bytes(b"model-bytes")
    return path


def test_publish_then_resolve_returns_the_file(config, model_file):
    store = ArtifactStore(config, code_version="test@v1")

    ref = store.publish(model_file, "xgboost_model", "cs2", params={"depth": 5}, metrics={"brier": 0.2})

    assert open(store.resolve(ref), "rb").read() == b"model-bytes"
    assert open(store.resolve(str(ref)), "rb").read() == b"model-bytes"
    assert store.latest("xgboost_model", "cs2") == ref


def test_versions_sort_by_publish_time(config, model_file):
    store = ArtifactStore(config)

    first = store.publish(model_file, "xgboost_model", "cs2")
    second = store.publish(model_file, "xgboost_model", "cs2")

    assert first.version < second.version
    assert store.list("xgboost_model", "cs2") == [first, second]


def test_alias_newest_row_wins_so_rollback_is_another_append(config, model_file):
    store = ArtifactStore(config)
    v1 = store.publish(model_file, "xgboost_model", "cs2")
    v2 = store.publish(model_file, "xgboost_model", "cs2")

    store.set_alias(v1, "production")
    store.set_alias(v2, "production")
    assert store.alias("xgboost_model", "cs2", "production") == v2

    store.set_alias(v1, "production")
    assert store.find(ArtifactQuery.parse("artifact://xgboost_model/cs2@production")) == v1


def test_set_alias_rejects_unknown_version(config, model_file):
    store = ArtifactStore(config)
    store.publish(model_file, "xgboost_model", "cs2")

    with pytest.raises(KeyError):
        store.set_alias(store.latest("xgboost_model", "cs2").__class__("xgboost_model", "cs2", "nope"), "production")


def test_inputs_record_what_the_model_was_trained_on_and_rebuild_matches(config, model_file):
    data = DatasetStore(config)
    data.publish(SPEC, pd.DataFrame({"match_id": [1, 2], "x": [0.1, 0.2]}))
    as_of = datetime.now(timezone.utc)
    data.load("priors", as_of=as_of)
    store = ArtifactStore(config)

    ref = store.publish(model_file, "xgboost_model", "cs2", inputs=data.loaded)
    data.publish(SPEC, pd.DataFrame({"match_id": [1, 2], "x": [0.5, 0.2]}))
    recorded = store.inputs(ref).iloc[0]
    rebuild = DatasetStore(config)
    rebuild.load("priors", as_of=recorded.as_of.to_pydatetime())

    assert recorded.dataset == "priors" and recorded.row_count == 2
    assert rebuild.loaded[0].rows_hash == recorded.rows_hash


def test_missing_artifact_raises_key_error(config):
    with pytest.raises(KeyError):
        ArtifactStore(config).latest("xgboost_model", "none")


@pytest.mark.parametrize("text,expected", [
    ("artifact://t/n", ArtifactQuery("t", "n")),
    ("artifact://t/n:20261008T000000000000Z", ArtifactQuery("t", "n", version="20261008T000000000000Z")),
    ("artifact://t/n@production", ArtifactQuery("t", "n", alias="production")),
])
def test_query_parsing(text, expected):
    assert ArtifactQuery.parse(text) == expected


def test_query_rejects_malformed():
    with pytest.raises(ValueError):
        ArtifactQuery.parse("artifact://only-type")
