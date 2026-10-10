"""Versioned files (models and other blobs) indexed in Iceberg.

Files live under CatalogConfig.artifact_root; three append-only tables index them:
`artifacts` (one row per version), `artifact_aliases` (newest row per alias wins,
so rollback is appending the previous version) and `artifact_inputs` (the datasets
each version was built from, so its inputs can be rebuilt with load(as_of=...)).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
from pyiceberg.exceptions import NoSuchTableError
from pyiceberg.expressions import And, BooleanExpression, EqualTo

from gnomepy._fs import fs_read_bytes, fs_write_bytes, resolve_fs
from gnomepy.data.catalog import CatalogConfig
from gnomepy.data.datasets import DatasetStore, InputRecord, TableSpec

ARTIFACTS = TableSpec(
    "artifacts",
    pa.schema([
        pa.field("artifact_type", pa.string(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("version", pa.string(), nullable=False),
        pa.field("uri", pa.string(), nullable=False),
        pa.field("file_format", pa.string()),
        pa.field("size_bytes", pa.int64()),
        pa.field("description", pa.string()),
        pa.field("params", pa.string()),
        pa.field("metrics", pa.string()),
        pa.field("run_id", pa.string()),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]),
    key=("artifact_type", "name", "version"),
)
ALIASES = TableSpec(
    "artifact_aliases",
    pa.schema([
        pa.field("artifact_type", pa.string(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("alias", pa.string(), nullable=False),
        pa.field("version", pa.string(), nullable=False),
        pa.field("set_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]),
    key=("artifact_type", "name", "alias"),
)
INPUTS = TableSpec(
    "artifact_inputs",
    pa.schema([
        pa.field("artifact_type", pa.string(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("version", pa.string(), nullable=False),
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("as_of", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("snapshot_id", pa.int64()),
        pa.field("row_count", pa.int64(), nullable=False),
        pa.field("rows_hash", pa.string(), nullable=False),
    ]),
    key=("artifact_type", "name", "version", "dataset"),
)


@dataclass(frozen=True)
class ArtifactRef:
    artifact_type: str
    name: str
    version: str
    uri: str = ""

    def __str__(self) -> str:
        return f"artifact://{self.artifact_type}/{self.name}:{self.version}"


@dataclass(frozen=True)
class ArtifactQuery:
    """A parsed artifact:// reference: a pinned version, an alias, or (neither) the latest."""

    artifact_type: str
    name: str
    version: str | None = None
    alias: str | None = None

    @classmethod
    def parse(cls, ref: str) -> ArtifactQuery:
        """artifact://type/name, artifact://type/name:VERSION or artifact://type/name@ALIAS."""
        if not ref.startswith("artifact://"):
            raise ValueError(f"not an artifact:// reference: {ref!r}")
        body = ref[len("artifact://"):]
        version = alias = None
        if "@" in body:
            body, alias = body.rsplit("@", 1)
        elif ":" in body:
            body, version = body.rsplit(":", 1)
        parts = body.split("/", 1)
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"expected artifact://type/name[:version|@alias], got {ref!r}")
        return cls(parts[0], parts[1], version, alias)


class ArtifactStore:
    def __init__(self, config: CatalogConfig | None = None, *, code_version: str = "unknown",
                 run_id: str | None = None, store: DatasetStore | None = None) -> None:
        self.config = config or CatalogConfig.from_env()
        self.run_id = run_id
        self._tables = store or DatasetStore(
            self.config, namespace=self.config.artifact_namespace, code_version=code_version, run_id=run_id
        )

    def publish(
        self,
        local_path: str | Path,
        artifact_type: str,
        name: str,
        *,
        description: str = "",
        params: dict | None = None,
        metrics: dict | None = None,
        inputs: list[InputRecord] | tuple[InputRecord, ...] = (),
    ) -> ArtifactRef:
        local_path = Path(local_path)
        created_at = datetime.now(timezone.utc)
        # Timestamp versions: concurrent Iceberg appends don't conflict, so an integer
        # counter read-then-incremented could hand two publishers the same number.
        version = created_at.strftime("%Y%m%dT%H%M%S%fZ")
        ext = local_path.suffix.lstrip(".")
        uri = f"{self.config.artifact_root.rstrip('/')}/{artifact_type}/{name}/{version}/artifact" + (f".{ext}" if ext else "")
        _write_file(uri, local_path.read_bytes())

        self._tables.publish(ARTIFACTS, pd.DataFrame([{
            "artifact_type": artifact_type, "name": name, "version": version, "uri": uri,
            "file_format": ext or "bin", "size_bytes": local_path.stat().st_size, "description": description,
            "params": json.dumps(params or {}, sort_keys=True, default=str),
            "metrics": json.dumps(metrics or {}, sort_keys=True, default=str),
            "run_id": self.run_id, "created_at": created_at,
        }]))
        if inputs:
            self._tables.publish(INPUTS, pd.DataFrame([{
                "artifact_type": artifact_type, "name": name, "version": version, "dataset": r.dataset,
                "as_of": r.as_of, "snapshot_id": r.snapshot_id, "row_count": r.row_count, "rows_hash": r.rows_hash,
            } for r in inputs]))
        return ArtifactRef(artifact_type, name, version, uri)

    def list(self, artifact_type: str | None = None, name: str | None = None) -> list[ArtifactRef]:
        conditions = [EqualTo(c, v) for c, v in (("artifact_type", artifact_type), ("name", name)) if v is not None]
        rows = self._load(ARTIFACTS.name, _all_of(conditions))
        return [ArtifactRef(r.artifact_type, r.name, r.version, r.uri)
                for r in rows.sort_values(["artifact_type", "name", "version"]).itertuples()]

    def get(self, artifact_type: str, name: str, version: str) -> ArtifactRef:
        rows = self._load(ARTIFACTS.name, _all_of([EqualTo("artifact_type", artifact_type), EqualTo("name", name),
                                                    EqualTo("version", version)]))
        if rows.empty:
            raise KeyError(f"no artifact {artifact_type}/{name}:{version}")
        r = rows.iloc[0]
        return ArtifactRef(r.artifact_type, r["name"], r.version, r.uri)

    def latest(self, artifact_type: str, name: str) -> ArtifactRef:
        refs = self.list(artifact_type, name)
        if not refs:
            raise KeyError(f"no artifact {artifact_type}/{name}")
        return refs[-1]

    def alias(self, artifact_type: str, name: str, alias: str) -> ArtifactRef:
        rows = self._load(ALIASES.name, _all_of([EqualTo("artifact_type", artifact_type), EqualTo("name", name),
                                                   EqualTo("alias", alias)]))
        if rows.empty:
            raise KeyError(f"no alias {artifact_type}/{name}@{alias}")
        return self.get(artifact_type, name, rows.iloc[0].version)

    def set_alias(self, ref: ArtifactRef, alias: str) -> None:
        self.get(ref.artifact_type, ref.name, ref.version)
        self._tables.publish(ALIASES, pd.DataFrame([{
            "artifact_type": ref.artifact_type, "name": ref.name, "alias": alias, "version": ref.version,
            "set_at": datetime.now(timezone.utc),
        }]))

    def inputs(self, ref: ArtifactRef) -> pd.DataFrame:
        return self._load(INPUTS.name, _all_of([EqualTo("artifact_type", ref.artifact_type), EqualTo("name", ref.name),
                                                 EqualTo("version", ref.version)]))

    def find(self, query: ArtifactQuery) -> ArtifactRef:
        if query.alias is not None:
            return self.alias(query.artifact_type, query.name, query.alias)
        if query.version is not None:
            return self.get(query.artifact_type, query.name, query.version)
        return self.latest(query.artifact_type, query.name)

    def resolve(self, ref: str | ArtifactRef) -> str:
        """A local path to the artifact's file, downloading it into the cache if needed."""
        if isinstance(ref, str):
            if not ref.startswith("artifact://"):
                return _download(ref) if ref.startswith("s3://") else ref
            ref = self.find(ArtifactQuery.parse(ref))
        local = _cache_base() / "artifacts" / ref.artifact_type / ref.name / ref.version / Path(ref.uri).name
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(_read_file(ref.uri))
        return str(local)

    def _load(self, table: str, condition: BooleanExpression | None) -> pd.DataFrame:
        try:
            return self._tables.load(table, row_filter=condition)
        except NoSuchTableError:
            return pd.DataFrame(columns=_spec(table).schema.names)


def resolve_artifact_path(path: str, *, config: CatalogConfig | None = None) -> str:
    """artifact://type/name[:version|@alias] or s3:// → a local cached file; anything else is returned as-is."""
    if path.startswith("artifact://"):
        return ArtifactStore(config).resolve(path)
    if path.startswith("s3://"):
        return _download(path)
    return path


def _spec(table: str) -> TableSpec:
    return {s.name: s for s in (ARTIFACTS, ALIASES, INPUTS)}[table]


def _all_of(conditions: list[BooleanExpression]) -> BooleanExpression | None:
    if not conditions:
        return None
    combined = conditions[0]
    for c in conditions[1:]:
        combined = And(combined, c)
    return combined


def _cache_base() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "gnomepy"


def _write_file(uri: str, data: bytes) -> None:
    fs, path = resolve_fs(uri)
    if not uri.startswith("s3://"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    fs_write_bytes(fs, path, data)


def _read_file(uri: str) -> bytes:
    fs, path = resolve_fs(uri)
    return fs_read_bytes(fs, path)


def _download(s3_uri: str) -> str:
    local = _cache_base() / "s3" / s3_uri.split("/", 3)[-1]
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(_read_file(s3_uri))
    return str(local)
