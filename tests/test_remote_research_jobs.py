"""The research job client: input sets, request bodies and downloads, with the controller and S3 stood in for."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gnomepy import remote

# The controller's inputs Lambda names the set {a.txt: "hello", sub/b.txt: "world"} this; the client must agree.
CONTROLLER_DIGEST = "0e4c56640c53e6f6bed2692aa3d6bb377ac3ad0fcf95ffccaba5a90b95ea41e3"


@pytest.fixture
def input_dir(tmp_path: Path) -> Path:
    (tmp_path / "set" / "sub").mkdir(parents=True)
    (tmp_path / "set" / "a.txt").write_text("hello")
    (tmp_path / "set" / "sub" / "b.txt").write_text("world")
    return tmp_path / "set"


def test_a_directory_is_named_as_the_controller_names_it(input_dir):
    manifest = remote.research_input_manifest([input_dir])
    assert [f["path"] for f in manifest] == ["a.txt", "sub/b.txt"]
    assert remote.research_input_digest(manifest) == CONTROLLER_DIGEST


def test_files_go_in_at_their_names_and_may_not_collide(input_dir, tmp_path):
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "a.txt").write_text("again")
    manifest = remote.research_input_manifest([input_dir / "a.txt", input_dir / "sub" / "b.txt"])
    assert [f["path"] for f in manifest] == ["a.txt", "b.txt"]
    with pytest.raises(ValueError, match="a.txt"):
        remote.research_input_manifest([input_dir / "a.txt", tmp_path / "other" / "a.txt"])


class Uploads:
    def __init__(self, exists: bool):
        self.exists = exists
        self.requests: list[tuple[str, str, dict]] = []
        self.puts: list[tuple[str, bytes]] = []

    def request(self, method, path, **kwargs):
        self.requests.append((method, path, kwargs))
        if path == "/research-jobs/inputs":
            files = kwargs["json"]["files"]
            if self.exists:
                return {"exists": True, "sha256": "d" * 64}
            return {"exists": False, "sha256": "d" * 64,
                    "uploads": {f["path"]: f"https://put/{f['path']}" for f in files},
                    "manifest_upload": "https://put/manifest.json"}
        return {"run_id": "r1"}

    def put(self, url, data):
        self.puts.append((url, data.read() if hasattr(data, "read") else data))

        class Ok:
            def raise_for_status(self):
                pass
        return Ok()


def test_a_new_set_uploads_every_file_then_the_manifest(input_dir, monkeypatch):
    fake = Uploads(exists=False)
    monkeypatch.setattr(remote, "_request", fake.request)
    monkeypatch.setattr(remote.requests, "put", fake.put)
    assert remote.upload_research_input([input_dir]) == "d" * 64
    assert [url for url, _ in fake.puts] == ["https://put/a.txt", "https://put/sub/b.txt", "https://put/manifest.json"]
    assert fake.puts[0][1] == b"hello"
    assert [f["path"] for f in json.loads(fake.puts[-1][1])] == ["a.txt", "sub/b.txt"]


def test_a_set_already_uploaded_is_not_sent_again(input_dir, monkeypatch):
    fake = Uploads(exists=True)
    monkeypatch.setattr(remote, "_request", fake.request)
    monkeypatch.setattr(remote.requests, "put", fake.put)
    assert remote.upload_research_input([input_dir]) == "d" * 64
    assert fake.puts == []


def test_submit_sends_only_what_was_given(monkeypatch):
    fake = Uploads(exists=True)
    monkeypatch.setattr(remote, "_request", fake.request)
    remote.submit_research_job("pkg.mod", "a" * 40, array=[["x"], ["y"]], inputs={"data": "d" * 64}, cpu=4)
    method, path, kwargs = fake.requests[-1]
    assert (method, path) == ("POST", "/research-jobs")
    assert kwargs["json"] == {"module": "pkg.mod", "research_commit": "a" * 40, "array": [["x"], ["y"]],
                              "inputs": {"data": "d" * 64}, "cpu": 4}


class FakeS3:
    KEYS = [
        "research-jobs/runs/r1/jobs/0/task.json",
        "research-jobs/runs/r1/jobs/0/log.txt",
        "research-jobs/runs/r1/jobs/0/out/prices_2025-09.parquet",
        "research-jobs/runs/r1/jobs/1/task.json",
        "research-jobs/runs/r1/jobs/1/log.txt",
        "research-jobs/runs/r1/jobs/1/out/prices_2025-10.parquet",
    ]

    def __init__(self, keys=KEYS):
        self.keys = keys

    def get_paginator(self, name):
        keys = self.keys

        class Paginator:
            def paginate(self, Bucket, Prefix):
                return [{"Contents": [{"Key": k} for k in keys if k.startswith(Prefix)]}]
        return Paginator()

    def download_file(self, bucket, key, local):
        Path(local).write_text(key)


@pytest.fixture
def run(monkeypatch):
    monkeypatch.setattr(remote, "get_research_job",
                        lambda run_id: {"output_prefix": "s3://gnome-research-dev/research-jobs/runs/r1/"})


def test_download_keeps_each_task_apart_by_default(run, monkeypatch, tmp_path):
    monkeypatch.setattr(remote.boto3, "client", lambda name: FakeS3())
    assert remote.download_research_job("r1", tmp_path) == 4
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()) == [
        "0/log.txt", "0/out/prices_2025-09.parquet", "1/log.txt", "1/out/prices_2025-10.parquet"]


def test_a_flat_download_merges_outputs_and_refuses_collisions(run, monkeypatch, tmp_path):
    monkeypatch.setattr(remote.boto3, "client", lambda name: FakeS3())
    assert remote.download_research_job("r1", tmp_path / "flat", flat=True) == 2
    assert sorted(p.name for p in (tmp_path / "flat").iterdir()) == ["prices_2025-09.parquet", "prices_2025-10.parquet"]

    clash = [k.replace("2025-10", "2025-09") for k in FakeS3.KEYS]
    monkeypatch.setattr(remote.boto3, "client", lambda name: FakeS3(clash))
    with pytest.raises(ValueError, match="tasks 0 and 1"):
        remote.download_research_job("r1", tmp_path / "clash", flat=True)
