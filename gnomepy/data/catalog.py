from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import boto3
from pyiceberg.catalog import Catalog, load_catalog

S3_TABLES_ENDPOINT = "https://s3tables.{region}.amazonaws.com/iceberg"


@dataclass(frozen=True)
class CatalogConfig:
    """Where tables and artifact files live.

    table_bucket_arn selects an S3 Tables bucket; local_warehouse selects a SQLite
    catalog on disk instead (tests and offline work). artifact_root is the S3 URI or
    local directory that artifact files are written under.
    """

    table_bucket_arn: str | None = None
    local_warehouse: Path | None = None
    namespace: str = "research"
    artifact_namespace: str = "artifacts"
    artifact_root: str = ""
    region: str = "us-east-1"

    @classmethod
    def from_env(cls) -> CatalogConfig:
        local = os.environ.get("GNOME_LOCAL_WAREHOUSE")
        stage = os.environ.get("STAGE", "prod").lower()
        return cls(
            table_bucket_arn=os.environ.get("GNOME_TABLE_BUCKET_ARN"),
            local_warehouse=Path(local) if local else None,
            namespace=os.environ.get("GNOME_TABLES_NAMESPACE", "research"),
            artifact_namespace=os.environ.get("GNOME_ARTIFACTS_NAMESPACE", "artifacts"),
            artifact_root=os.environ.get("GNOME_ARTIFACT_ROOT", f"s3://gnome-research-{stage}/artifacts"),
            region=os.environ.get("AWS_REGION", "us-east-1"),
        )


def open_catalog(config: CatalogConfig) -> Catalog:
    if config.local_warehouse is not None:
        config.local_warehouse.mkdir(parents=True, exist_ok=True)
        return load_catalog(
            "local",
            type="sql",
            uri=f"sqlite:///{config.local_warehouse}/catalog.db",
            warehouse=f"file://{config.local_warehouse}",
        )
    if config.table_bucket_arn:
        return load_catalog(
            "s3tables",
            type="rest",
            uri=S3_TABLES_ENDPOINT.format(region=config.region),
            warehouse=config.table_bucket_arn,
            **{
                "rest.sigv4-enabled": "true",
                "rest.signing-name": "s3tables",
                "rest.signing-region": config.region,
                "s3.region": config.region,
                "s3.connect-timeout": "10",
                "s3.request-timeout": "60",
                **_sso_credentials(),
            },
        )
    raise ValueError("CatalogConfig needs table_bucket_arn or local_warehouse (set GNOME_TABLE_BUCKET_ARN)")


def _sso_credentials() -> dict[str, str]:
    # pyarrow's S3 client can't read sso_session profiles, so a laptop's uploads go out
    # unsigned. Instance and container roles are left to pyarrow: frozen keys would expire
    # under a long-running session, while pyarrow refreshes role credentials itself.
    credentials = boto3.Session().get_credentials()
    if credentials is None or credentials.method != "sso":
        return {}
    frozen = credentials.get_frozen_credentials()
    return {
        "s3.access-key-id": frozen.access_key,
        "s3.secret-access-key": frozen.secret_key,
        "s3.session-token": frozen.token,
    }
