from pathlib import Path

from gnomepy.data import CatalogConfig


def test_env_vars_configure_s3_tables(monkeypatch):
    monkeypatch.delenv("GNOME_LOCAL_WAREHOUSE", raising=False)
    monkeypatch.setenv("GNOME_TABLE_BUCKET_ARN", "arn:aws:s3tables:us-east-1:1:bucket/b")
    monkeypatch.setenv("GNOME_ARTIFACT_ROOT", "s3://bucket/artifacts")
    monkeypatch.setenv("AWS_REGION", "us-east-1")

    config = CatalogConfig.from_env()

    assert config.table_bucket_arn == "arn:aws:s3tables:us-east-1:1:bucket/b"
    assert config.artifact_root == "s3://bucket/artifacts"


def test_local_warehouse_keeps_artifact_files_beside_it(monkeypatch, tmp_path):
    monkeypatch.delenv("GNOME_TABLE_BUCKET_ARN", raising=False)
    monkeypatch.delenv("GNOME_ARTIFACT_ROOT", raising=False)
    monkeypatch.setenv("GNOME_LOCAL_WAREHOUSE", str(tmp_path))

    config = CatalogConfig.from_env()

    assert config.local_warehouse == Path(tmp_path)
    assert config.artifact_root == str(tmp_path / "files")
