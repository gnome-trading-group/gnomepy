from __future__ import annotations

import tempfile
from pathlib import Path

import boto3

from gnomepy.signals.registrable import Registrable


class SignalRegistry:
    """Save and load signal models to/from S3.

    S3 key prefix: signals/{name}/{version}/

    Usage:
        registry = SignalRegistry(bucket="gnome-signals")
        registry.save(my_model)
        registry.load(my_model, "ewma-volatility", "v1.0")
    """

    PREFIX = "signals/"

    def __init__(self, bucket: str = "gnome-signals", s3_client=None):
        self._bucket = bucket
        self._s3 = s3_client or boto3.client("s3")

    def save(self, signal: Registrable):
        """Save model to S3: model writes to temp dir, registry uploads."""
        with tempfile.TemporaryDirectory(prefix="signal-save-") as tmp:
            tmp_path = Path(tmp)
            signal.save_model(tmp_path)

            s3_prefix = self.PREFIX + signal.get_name() + "/" + signal.get_version() + "/"
            for file in tmp_path.rglob("*"):
                if file.is_file():
                    key = s3_prefix + str(file.relative_to(tmp_path))
                    self._s3.upload_file(str(file), self._bucket, key)

    def load(self, signal: Registrable, name: str, version: str):
        """Load model from S3: registry downloads to temp dir, model reads."""
        with tempfile.TemporaryDirectory(prefix="signal-load-") as tmp:
            tmp_path = Path(tmp)

            s3_prefix = self.PREFIX + name + "/" + version + "/"
            response = self._s3.list_objects_v2(Bucket=self._bucket, Prefix=s3_prefix)

            for obj in response.get("Contents", []):
                relative = obj["Key"][len(s3_prefix):]
                local_file = tmp_path / relative
                local_file.parent.mkdir(parents=True, exist_ok=True)
                self._s3.download_file(self._bucket, obj["Key"], str(local_file))

            signal.load_model(tmp_path)

    def list_versions(self, name: str) -> list[str]:
        """List available versions for a model name."""
        prefix = self.PREFIX + name + "/"
        response = self._s3.list_objects_v2(Bucket=self._bucket, Prefix=prefix, Delimiter="/")

        versions = []
        for cp in response.get("CommonPrefixes", []):
            version = cp["Prefix"][len(prefix):].rstrip("/")
            versions.append(version)
        return sorted(versions)
