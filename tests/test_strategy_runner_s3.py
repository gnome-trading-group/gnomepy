import subprocess
import sys

FINALIZE_WARNING = "FinalizeS3 was not called"

# A session ends through System.exit, so Python's atexit (pyarrow's own S3 finalizer) never runs.
_EXIT_THROUGH_SYSTEM_EXIT = """
import sys
import jpype
import pyarrow.fs as pafs
from gnomepy.java.strategy.runner import _register_s3_finalizer

mode = sys.argv[1]
jpype.startJVM()
if mode != "no-s3":
    pafs.S3FileSystem(anonymous=True, region="us-east-1")
if mode != "no-hook":
    _register_s3_finalizer()
jpype.JClass("java.lang.System").exit(0)
"""


def _exit_through_system_exit(mode: str, tmp_path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _EXIT_THROUGH_SYSTEM_EXIT, mode],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )


def test_system_exit_without_the_finalizer_warns(tmp_path):
    assert FINALIZE_WARNING in _exit_through_system_exit("no-hook", tmp_path).stderr


def test_system_exit_with_the_finalizer_is_clean(tmp_path):
    result = _exit_through_system_exit("hook", tmp_path)

    assert FINALIZE_WARNING not in result.stderr
    assert result.returncode == 0


def test_finalizer_does_nothing_when_s3_was_never_initialized(tmp_path):
    result = _exit_through_system_exit("no-s3", tmp_path)

    assert result.stderr == ""
    assert result.returncode == 0
