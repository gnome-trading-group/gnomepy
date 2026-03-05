"""Shared CLI utilities for loading YAML configs from local paths or S3."""

from __future__ import annotations

import tempfile

import yaml


def load_yaml(path: str) -> dict:
    """Load a YAML file from a local path or S3 URI, returning the parsed dict."""
    if path.startswith("s3://"):
        raw = _download_s3(path)
    else:
        with open(path) as f:
            raw = f.read()

    return yaml.safe_load(raw)


def _download_s3(uri: str) -> str:
    import boto3

    parts = uri.replace("s3://", "").split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 URI: {uri}")
    bucket, key = parts

    s3 = boto3.client("s3")
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=True) as tmp:
        s3.download_file(bucket, key, tmp.name)
        with open(tmp.name) as f:
            return f.read()
