"""HTTP client for the gnome-controller backtest and research job APIs (API Gateway + Cognito auth)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import boto3
import requests

from gnomepy.auth import get_id_token
from gnomepy.config import config


def _api_base_url() -> str:
    return os.environ.get("GNOME_CONTROLLER_API_URL", config.CONTROLLER_API_URL).rstrip("/")


def _headers() -> dict[str, str]:
    return {
        "Authorization": get_id_token(),
        "Content-Type": "application/json",
    }


def _request(method: str, path: str, **kwargs) -> dict:
    url = f"{_api_base_url()}{path}"
    resp = requests.request(method, url, headers=_headers(), **kwargs)
    if resp.status_code >= 400:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise RuntimeError(f"API error {resp.status_code}: {detail}")
    return resp.json()


def submit_backtest(config_yaml: str, research_commit: str = "main") -> dict:
    return _request("POST", "/backtests", json={
        "config": config_yaml,
        "research_commit": research_commit,
    })


def get_backtest(run_id: str) -> dict:
    return _request("GET", f"/backtests/{run_id}")


def list_backtests(status: str | None = None, limit: int = 20) -> dict:
    params: dict = {"limit": str(limit)}
    if status:
        params["status"] = status
    return _request("GET", "/backtests", params=params)


def cancel_backtest(run_id: str) -> dict:
    return _request("DELETE", f"/backtests/{run_id}")


def research_input_manifest(paths: list[Path]) -> list[dict]:
    """The files of an input set, each with its path inside the set, size and sha256.

    A directory contributes every file under it, at its path relative to the directory; a file goes in at its name.
    """
    files: dict[str, Path] = {}
    for path in paths:
        entries = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for entry in entries:
            name = entry.relative_to(path).as_posix() if path.is_dir() else entry.name
            if name in files:
                raise ValueError(f"two input files would both be {name!r}")
            files[name] = entry
    if not files:
        raise ValueError("no input files found")
    return [{"path": name, "size": f.stat().st_size, "sha256": _sha256(f), "local": f} for name, f in sorted(files.items())]


def research_input_digest(manifest: list[dict]) -> str:
    """The input set's name; the controller computes the same digest from the same manifest."""
    entries = sorted(({k: f[k] for k in ("path", "size", "sha256")} for f in manifest), key=lambda f: f["path"])
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def upload_research_input(paths: list[Path]) -> str:
    """Upload an input set once (content-addressed) and return the sha256 a job refers to it by."""
    manifest = research_input_manifest(paths)
    files = [{k: f[k] for k in ("path", "size", "sha256")} for f in manifest]
    response = _request("POST", "/research-jobs/inputs", json={"files": files})
    if response["exists"]:
        return response["sha256"]
    for f in manifest:
        with f["local"].open("rb") as body:
            requests.put(response["uploads"][f["path"]], data=body).raise_for_status()
    # Last, because its presence is what marks the set complete.
    requests.put(response["manifest_upload"], data=json.dumps(files, indent=2).encode()).raise_for_status()
    return response["sha256"]


def submit_research_job(
    module: str,
    research_commit: str,
    args: list[str] | None = None,
    array: list[list[str]] | None = None,
    inputs: dict[str, str] | None = None,
    cpu: int | None = None,
    memory_gib: int | None = None,
    research_ref: str | None = None,
    description: str | None = None,
) -> dict:
    """Run `python -m <module> <args>` at a gnomepy-research commit: once with args, or once per entry of array.

    inputs maps a name to an uploaded input set's sha256 (upload_research_input); each task sees it at
    /work/in/<name>/ and writes its outputs to $OUTPUT_DIR.
    """
    body: dict = {"module": module, "research_commit": research_commit}
    optional = {"args": args, "array": array, "inputs": inputs, "cpu": cpu, "memory_gib": memory_gib,
                "research_ref": research_ref, "description": description}
    body.update({k: v for k, v in optional.items() if v is not None})
    return _request("POST", "/research-jobs", json=body)


def get_research_job(run_id: str) -> dict:
    return _request("GET", f"/research-jobs/{run_id}")


def list_research_jobs(status: str | None = None, limit: int = 20) -> dict:
    params: dict = {"limit": str(limit)}
    if status:
        params["status"] = status
    return _request("GET", "/research-jobs", params=params)


def cancel_research_job(run_id: str) -> dict:
    return _request("DELETE", f"/research-jobs/{run_id}")


def download_research_job(run_id: str, dest: Path, flat: bool = False) -> int:
    """Copy a run's outputs with the caller's AWS credentials; returns the file count.

    By default each task's outputs and log go to dest/<task index>/{out/,log.txt}. flat merges every task's
    outputs into dest itself (no logs), for runs whose tasks write distinct files, and refuses a name two tasks wrote.
    """
    run = get_research_job(run_id)
    bucket, prefix = run["output_prefix"].removeprefix("s3://").split("/", 1)
    jobs_prefix = f"{prefix}jobs/"
    s3 = boto3.client("s3")
    targets: dict[Path, tuple[str, str]] = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=jobs_prefix):
        for obj in page.get("Contents", []):
            index, _, rest = obj["Key"][len(jobs_prefix):].partition("/")
            if rest == "task.json":
                continue
            if flat:
                if not rest.startswith("out/"):
                    continue
                local = dest / rest.removeprefix("out/")
                if local in targets:
                    raise ValueError(f"tasks {targets[local][1]} and {index} both wrote {local.relative_to(dest)}")
            else:
                local = dest / index / rest
            targets[local] = (obj["Key"], index)
    for local, (key, _) in targets.items():
        local.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(local))
    return len(targets)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
