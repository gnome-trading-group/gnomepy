"""
Run one task of a research job: `python -m <module> <args>` at a pinned gnomepy-research commit.

The task spec is s3://$S3_BUCKET/research-jobs/runs/$RUN_ID/jobs/<index>/task.json: {module, args, inputs}, where
inputs maps a name to an uploaded input set's S3 prefix. The task sees each input set at /work/in/<name>/ and writes
its outputs to $OUTPUT_DIR (/work/out). Whatever happens, the outputs and the task's log are uploaded to
.../jobs/<index>/out/ and .../jobs/<index>/log.txt, and the task's exit status is the container's.

Env:
    RUN_ID, RESEARCH_COMMIT                  per run (the submit Lambda)
    S3_BUCKET, AWS_DEFAULT_REGION            the job definition
    AWS_BATCH_JOB_ARRAY_INDEX                Batch, for array jobs; else JOB_INDEX, else 0

The GitHub token for the private repo comes from Secrets Manager (gnomepy/gh-token) and never reaches the task.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import tomllib
import traceback
from pathlib import Path
from typing import IO

import boto3

WORK = Path("/work")
OUTPUT_DIR = WORK / "out"
INPUT_DIR = WORK / "in"
LOG = WORK / "log.txt"
REPO = Path("/opt/gnomepy-research")
REPO_URL = "https://x-access-token:{token}@github.com/gnome-trading-group/gnomepy-research.git"

_child: subprocess.Popen | None = None
_terminated = False


def main() -> int:
    run_id, bucket, commit = _env("RUN_ID"), _env("S3_BUCKET"), _env("RESEARCH_COMMIT")
    _env("AWS_DEFAULT_REGION")
    index = os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX") or os.environ.get("JOB_INDEX") or "0"
    prefix = f"research-jobs/runs/{run_id}/jobs/{index}/"
    s3 = boto3.client("s3")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _on_sigterm)

    status = 1
    with LOG.open("w") as log:
        try:
            status = _run(s3, bucket, prefix, commit, index, log)
        except Exception:
            _say(log, traceback.format_exc())
        _say(log, f"run_task: exit status {status}; uploading outputs to s3://{bucket}/{prefix}out/")
    try:
        _upload(s3, bucket, prefix)
    except Exception:
        traceback.print_exc()
        status = status or 1
    return status


def _run(s3, bucket: str, prefix: str, commit: str, index: str, log: IO[str]) -> int:
    _say(log, f"run_task: {prefix} at gnomepy-research {commit}")
    task = json.loads(s3.get_object(Bucket=bucket, Key=f"{prefix}task.json")["Body"].read())

    for name, uri in task.get("inputs", {}).items():
        count = _download_prefix(s3, uri, INPUT_DIR / name)
        _say(log, f"run_task: input {name}: {count} files from {uri}")

    token = _github_token()
    clone = subprocess.run(["git", "clone", "--filter=blob:none", "--quiet", REPO_URL.format(token=token), str(REPO)],
                           capture_output=True, text=True)
    if clone.returncode != 0:
        # git echoes the URL it failed on; keep the token out of the log.
        _say(log, clone.stderr.replace(token, "***"))
        return clone.returncode
    if (code := _stream(["git", "-C", str(REPO), "checkout", "--quiet", commit], log)) != 0:
        return code
    _stream(["git", "-C", str(REPO), "--no-pager", "log", "-1", "--oneline"], log)

    # The image's gnomepy stays: a commit that needs a newer one fails here, by name, rather than at import.
    pin = WORK / "gnomepy-pin.txt"
    pin.write_text(f"gnomepy=={importlib.metadata.version('gnomepy')}\n")
    extras = ",".join(tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["optional-dependencies"])
    install = [sys.executable, "-m", "pip", "install", "--quiet", "-c", str(REPO / "constraints.txt"), "-c", str(pin),
               f"{REPO}[{extras}]"]
    if (code := _stream(install, log)) != 0:
        return code

    command = [sys.executable, "-m", task["module"], *task.get("args", [])]
    _say(log, f"run_task: {' '.join(command[1:])}")
    env = {**os.environ, "OUTPUT_DIR": str(OUTPUT_DIR), "INPUT_DIR": str(INPUT_DIR), "JOB_INDEX": index}
    return _stream(command, log, env=env, cwd=WORK)


def _stream(command: list[str], log: IO[str], env: dict | None = None, cwd: Path | None = None) -> int:
    """Run command, copying its output to both stdout (CloudWatch) and the task log."""
    global _child
    if _terminated:
        return 143
    _child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=cwd,
                              bufsize=1)
    for line in _child.stdout:
        sys.stdout.write(line)
        log.write(line)
    code = _child.wait()
    _child = None
    return code


def _on_sigterm(signum, frame) -> None:
    # Batch sends SIGTERM on cancel and timeout: stop the task, then fall through to the upload.
    global _terminated
    _terminated = True
    if _child is not None:
        _child.terminate()


def _say(log: IO[str], message: str) -> None:
    print(message, flush=True)
    log.write(message.rstrip("\n") + "\n")
    log.flush()


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"run_task: {name} is required")
    return value


def _github_token() -> str:
    value = boto3.client("secretsmanager").get_secret_value(SecretId="gnomepy/gh-token")["SecretString"]
    try:
        return json.loads(value)["token"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return value.strip()


def _download_prefix(s3, uri: str, dest: Path) -> int:
    bucket, prefix = uri.removeprefix("s3://").split("/", 1)
    count = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            relative = obj["Key"][len(prefix):]
            if relative == "manifest.json":
                continue
            local = dest / relative
            local.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, obj["Key"], str(local))
            count += 1
    return count


def _upload(s3, bucket: str, prefix: str) -> None:
    for path in sorted(p for p in OUTPUT_DIR.rglob("*") if p.is_file()):
        s3.upload_file(str(path), bucket, f"{prefix}out/{path.relative_to(OUTPUT_DIR).as_posix()}")
    s3.upload_file(str(LOG), bucket, f"{prefix}log.txt")


if __name__ == "__main__":
    sys.exit(main())
