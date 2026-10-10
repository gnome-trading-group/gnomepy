"""Append-only, point-in-time datasets stored as Iceberg tables.

Rows are never overwritten: a correction or re-scrape appends a new row with a later
ingested_at, and readers take the newest row per key. load(name, as_of=T) therefore
returns the table exactly as it was known at T, for any T, without depending on how
long the catalog keeps old snapshots.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
from pyiceberg.catalog import Catalog
from pyiceberg.exceptions import CommitFailedException, NoSuchTableError
from pyiceberg.expressions import AlwaysTrue, And, BooleanExpression, In, LessThanOrEqual
from pyiceberg.table import Table
from pyiceberg.transforms import DayTransform, IdentityTransform, MonthTransform, Transform

from gnomepy.data.catalog import CatalogConfig, open_catalog

INGESTED_AT = "ingested_at"
CODE_VERSION = "code_version"
ROW_HASH = "row_hash"
MANAGED_FIELDS = (
    pa.field(INGESTED_AT, pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field(CODE_VERSION, pa.string(), nullable=False),
    pa.field(ROW_HASH, pa.int64(), nullable=False),
)
MANAGED_COLUMNS = tuple(f.name for f in MANAGED_FIELDS)
KEY_PROPERTY = "gnome.key"
COMMIT_ATTEMPTS = 4
TRANSFORMS: dict[str, Transform] = {"identity": IdentityTransform(), "day": DayTransform(), "month": MonthTransform()}


@dataclass(frozen=True)
class TableSpec:
    """A dataset's declared schema, its key, and optionally how it is partitioned.

    The schema is declared rather than inferred from a frame: an all-null column
    infers as Arrow's null type, which Iceberg v2 cannot store.
    """

    name: str
    schema: pa.Schema
    key: tuple[str, ...]
    partition_by: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        names = set(self.schema.names)
        if not self.key:
            raise ValueError(f"{self.name}: key must name at least one column")
        missing = [k for k in self.key if k not in names]
        if missing:
            raise ValueError(f"{self.name}: key columns {missing} are not in the schema")
        clashes = names & set(MANAGED_COLUMNS)
        if clashes:
            raise ValueError(f"{self.name}: {sorted(clashes)} are managed by DatasetStore and can't be declared")
        for f in self.schema:
            if pa.types.is_timestamp(f.type) and f.type.unit != "us":
                raise ValueError(f"{self.name}.{f.name}: Iceberg stores microsecond timestamps; declare unit 'us'")
        if self.partition_by is not None:
            column, transform = self.partition_by
            if column not in names or transform not in TRANSFORMS:
                raise ValueError(f"{self.name}: bad partition_by {self.partition_by}")

    def full_schema(self) -> pa.Schema:
        return pa.schema([*self.schema, *MANAGED_FIELDS])


@dataclass(frozen=True)
class InputRecord:
    """What a load returned, so a model can record exactly which data it used."""

    dataset: str
    as_of: datetime
    snapshot_id: int | None
    row_count: int
    rows_hash: str


@dataclass(frozen=True)
class PublishResult:
    rows_written: int
    rows_skipped: int
    snapshot_id: int | None


class DatasetStore:
    """Publish and load append-only datasets in one Iceberg namespace.

    Every load is remembered in `loaded`, so a training run can hand the records to
    ArtifactStore.publish(inputs=...) without tracking them itself.
    """

    def __init__(
        self,
        config: CatalogConfig | None = None,
        *,
        namespace: str | None = None,
        code_version: str = "unknown",
        run_id: str | None = None,
        catalog: Catalog | None = None,
    ) -> None:
        self.config = config or CatalogConfig.from_env()
        self.namespace = namespace or self.config.namespace
        self.code_version = code_version
        self.run_id = run_id
        self.loaded: list[InputRecord] = []
        self._catalog = catalog or open_catalog(self.config)
        self._catalog.create_namespace_if_not_exists(self.namespace)

    def create(self, spec: TableSpec) -> Table:
        """Create the table if it doesn't exist; an existing table must already match the spec's columns."""
        ident = self._ident(spec.name)
        try:
            table = self._catalog.load_table(ident)
        except NoSuchTableError:
            table = self._catalog.create_table(
                ident, schema=spec.full_schema(), properties={KEY_PROPERTY: ",".join(spec.key)}
            )
            if spec.partition_by is not None:
                column, transform = spec.partition_by
                with table.update_spec() as update:
                    update.add_field(column, TRANSFORMS[transform], f"{column}_{transform}")
            return table
        existing = {f.name for f in table.schema().fields} - set(MANAGED_COLUMNS)
        declared = set(spec.schema.names)
        if existing != declared:
            raise ValueError(
                f"{spec.name}: table columns {sorted(existing ^ declared)} differ from the spec; call evolve() to change the schema"
            )
        return table

    def evolve(self, spec: TableSpec) -> Table:
        """Add the spec's new columns to an existing table. Existing rows read them as null."""
        table = self._catalog.load_table(self._ident(spec.name))
        with table.update_schema() as update:
            update.union_by_name(spec.full_schema())
        return table

    def publish(self, spec: TableSpec, df: pd.DataFrame, *, ingested_at: datetime | None = None) -> PublishResult:
        """Append the rows of df that differ from the newest stored row for their key."""
        table = self.create(spec)
        _check_columns(spec, df)
        frame = _to_microseconds(df[list(spec.schema.names)]).reset_index(drop=True)
        frame[ROW_HASH] = _row_hashes(frame)

        frame = frame.drop_duplicates([*spec.key, ROW_HASH])
        conflicting = frame[frame.duplicated(list(spec.key), keep=False)]
        if len(conflicting):
            raise ValueError(
                f"{spec.name}: {conflicting[list(spec.key)].drop_duplicates().shape[0]} keys appear more than once "
                f"with different values in one publish, e.g. {conflicting[list(spec.key)].iloc[0].to_dict()}"
            )

        changed = _changed_rows(table, spec, frame)
        skipped = len(df) - len(changed)
        if changed.empty:
            return PublishResult(0, skipped, _snapshot_id(table))

        stamp = (ingested_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        changed = changed.assign(**{INGESTED_AT: pd.Timestamp(stamp).as_unit("us"), CODE_VERSION: self.code_version})
        rows = pa.Table.from_pandas(changed[spec.full_schema().names], schema=spec.full_schema(), preserve_index=False)
        _append_with_retry(table, rows, self._snapshot_properties())
        return PublishResult(len(changed), skipped, _snapshot_id(table))

    def load(
        self,
        name: str,
        *,
        as_of: datetime | None = None,
        columns: list[str] | None = None,
        row_filter: BooleanExpression | None = None,
        include_metadata: bool = False,
    ) -> pd.DataFrame:
        """The newest row per key, as known at as_of (default now)."""
        table = self._catalog.load_table(self._ident(name))
        key = table.properties[KEY_PROPERTY].split(",")
        as_of = (as_of or datetime.now(timezone.utc)).astimezone(timezone.utc)

        condition = And(row_filter or AlwaysTrue(), LessThanOrEqual(INGESTED_AT, as_of.isoformat()))
        fields = ("*",) if columns is None else tuple(dict.fromkeys([*key, *columns, *MANAGED_COLUMNS]))
        snapshot_id = _snapshot_id(table)
        rows = table.scan(row_filter=condition, selected_fields=fields, snapshot_id=snapshot_id).to_pandas()
        rows = rows.sort_values(INGESTED_AT, kind="stable").drop_duplicates(key, keep="last")
        rows = rows.sort_values(key, kind="stable").reset_index(drop=True)

        self.loaded.append(InputRecord(name, as_of, snapshot_id, len(rows), _combined_hash(rows[ROW_HASH])))
        if include_metadata:
            return rows
        return rows.drop(columns=list(MANAGED_COLUMNS))

    def list(self) -> list[str]:
        return sorted(ident[-1] for ident in self._catalog.list_tables(self.namespace))

    def _ident(self, name: str) -> str:
        return f"{self.namespace}.{name}"

    def _snapshot_properties(self) -> dict[str, str]:
        props = {"gnome.code_version": self.code_version}
        if self.run_id:
            props["gnome.run_id"] = self.run_id
        return props


def _check_columns(spec: TableSpec, df: pd.DataFrame) -> None:
    declared, given = set(spec.schema.names), set(df.columns)
    if declared != given:
        raise ValueError(
            f"{spec.name}: frame columns don't match the spec "
            f"(missing {sorted(declared - given)}, unexpected {sorted(given - declared)}); call evolve() to add columns"
        )


def _to_microseconds(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        dtype = out[column].dtype
        if isinstance(dtype, pd.DatetimeTZDtype) or np.issubdtype(dtype, np.datetime64):
            out[column] = out[column].dt.as_unit("us")
    return out


def _row_hashes(frame: pd.DataFrame) -> pd.Series:
    hashable = frame.copy()
    for column in hashable.columns:
        if hashable[column].dtype == object and hashable[column].map(_is_list_like).any():
            hashable[column] = hashable[column].map(_list_to_json)
    return pd.util.hash_pandas_object(hashable, index=False).astype("uint64").astype("int64")


def _is_list_like(value: object) -> bool:
    return isinstance(value, (list, tuple, np.ndarray))


def _list_to_json(value: object) -> object:
    # pandas can't hash arrays, and Parquet hands list columns back as numpy arrays.
    return json.dumps(np.asarray(value).tolist()) if _is_list_like(value) else value


def _changed_rows(table: Table, spec: TableSpec, frame: pd.DataFrame) -> pd.DataFrame:
    if table.current_snapshot() is None:
        return frame
    first = spec.key[0]
    stored = table.scan(
        row_filter=In(first, set(frame[first].tolist())),
        selected_fields=(*spec.key, ROW_HASH, INGESTED_AT),
    ).to_pandas()
    newest = stored.sort_values(INGESTED_AT, kind="stable").drop_duplicates(list(spec.key), keep="last")
    merged = frame.merge(
        newest[[*spec.key, ROW_HASH]].rename(columns={ROW_HASH: "_stored_hash"}), on=list(spec.key), how="left"
    )
    return frame[(merged["_stored_hash"] != merged[ROW_HASH]).to_numpy()]


def _append_with_retry(table: Table, rows: pa.Table, properties: dict[str, str]) -> None:
    for attempt in range(1, COMMIT_ATTEMPTS + 1):
        try:
            table.append(rows, snapshot_properties=properties)
            return
        except CommitFailedException:
            # S3 Tables compaction commits concurrently with writers.
            if attempt == COMMIT_ATTEMPTS:
                raise
            time.sleep(0.5 * attempt)
            table.refresh()


def _snapshot_id(table: Table) -> int | None:
    snapshot = table.current_snapshot()
    return None if snapshot is None else snapshot.snapshot_id


def _combined_hash(hashes: pd.Series) -> str:
    return hashlib.sha256(np.sort(hashes.to_numpy(dtype="int64")).tobytes()).hexdigest()
