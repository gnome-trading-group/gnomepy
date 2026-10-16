import gzip

import pyarrow.fs as pafs

from gnomepy import _fs
from gnomepy._fs import fs_read_bytes, fs_write_bytes, fs_write_json


def test_bytes_round_trip_unchanged_whatever_the_suffix(tmp_path):
    fs = pafs.LocalFileSystem()
    page = gzip.compress(b"<html><footer></footer></html>")

    for name in ("page.html.gz", "data.zst", "plain.bin"):
        fs_write_bytes(fs, str(tmp_path / name), page)
        assert fs_read_bytes(fs, str(tmp_path / name)) == page


def test_json_is_written_as_text_even_under_a_compressed_suffix(tmp_path):
    fs = pafs.LocalFileSystem()

    fs_write_json(fs, str(tmp_path / "summary.json.gz"), {"a": 1})

    assert (tmp_path / "summary.json.gz").read_bytes().startswith(b"{")


def test_an_s3_client_talks_to_the_buckets_region_not_the_callers(monkeypatch):
    # A strategy session in eu-west-1 reads research artifacts from a us-east-1 bucket; its own region gets a 301.
    _fs._bucket_region.cache_clear()
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "y")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    monkeypatch.setattr(_fs.pafs, "resolve_s3_region", lambda bucket: {"research": "us-east-1"}[bucket])

    fs, path = _fs.resolve_fs("s3://research/artifacts/model/artifact.json")

    assert fs.region == "us-east-1" and path == "research/artifacts/model/artifact.json"


def test_an_s3_client_falls_back_to_the_callers_region_when_the_bucket_cant_be_located(monkeypatch):
    _fs._bucket_region.cache_clear()
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "y")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")

    def unreachable(bucket):
        raise OSError("no network")
    monkeypatch.setattr(_fs.pafs, "resolve_s3_region", unreachable)

    fs, _ = _fs.resolve_fs("s3://research/key")

    assert fs.region == "eu-west-1"
