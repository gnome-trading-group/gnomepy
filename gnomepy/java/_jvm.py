from __future__ import annotations

import atexit
import threading
from pathlib import Path

import jpype

from gnomepy.java._classpath import discover_classpath

_lock = threading.Lock()

# Required for Agrona's UnsafeBuffer used by SBE codecs
DEFAULT_JVM_ARGS = [
    "--add-opens=java.base/jdk.internal.misc=ALL-UNNAMED",
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
    "--add-opens=java.base/java.nio=ALL-UNNAMED",
    "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
]


def ensure_jvm_started(
    classpath: list[str] | None = None,
    jvm_path: str | None = None,
    jvm_args: list[str] | None = None,
    gnome_root: str | Path | None = None,
) -> None:
    """Start the JVM if not already running. Idempotent and thread-safe.

    Args:
        classpath: Explicit list of JAR paths. If None, auto-discovered.
        jvm_path: Path to the JVM shared library. If None, JPype finds it.
        jvm_args: Extra JVM arguments. Merged with DEFAULT_JVM_ARGS.
        gnome_root: Root directory of GNOME repos for JAR discovery.
    """
    if jpype.isJVMStarted():
        return

    with _lock:
        if jpype.isJVMStarted():
            return

        if classpath is None:
            classpath = discover_classpath(gnome_root=gnome_root)

        args = list(DEFAULT_JVM_ARGS)
        if jvm_args:
            args.extend(jvm_args)

        kwargs = {
            "classpath": classpath,
            "convertStrings": True,
        }
        if jvm_path:
            kwargs["jvmpath"] = jvm_path

        jpype.startJVM(*args, **kwargs)
        atexit.register(_shutdown_jvm_safe)


def shutdown_jvm() -> None:
    """Shutdown the JVM. Can only be called once per process."""
    if jpype.isJVMStarted():
        jpype.shutdownJVM()


def is_jvm_started() -> bool:
    """Check if the JVM is currently running."""
    return jpype.isJVMStarted()


def _shutdown_jvm_safe() -> None:
    """Safe shutdown for atexit — ignores errors."""
    try:
        if jpype.isJVMStarted():
            jpype.shutdownJVM()
    except Exception:
        pass


class JVMContext:
    """Context manager for JVM lifecycle.

    Usage:
        with JVMContext(gnome_root="/path/to/gnome"):
            # JVM is running here
            ...
        # JVM is shut down
    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def __enter__(self):
        ensure_jvm_started(**self._kwargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutdown_jvm()
        return False

