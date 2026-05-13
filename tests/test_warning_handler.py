"""Integration test for the Java WarningHandler → Python warnings chain.

Requires a JVM with gnome-backtest JARs on the classpath (GNOME_JARS env var
or the standard discovery path). Skipped automatically when not available.
"""
from __future__ import annotations

import pytest


def _jvm_available() -> bool:
    try:
        from gnomepy.java._classpath import discover_classpath
        cp = discover_classpath()
        return len(cp) > 0
    except Exception:
        return False


@pytest.mark.skipif(not _jvm_available(), reason="JVM/gnome-backtest JARs not available")
def test_warning_handler_captures_java_warnings():
    import jpype
    from gnomepy.java._jvm import ensure_jvm_started

    ensure_jvm_started()

    WarningHandler = jpype.JClass("group.gnometrading.backtest.recorder.WarningHandler")
    Logger = jpype.JClass("java.util.logging.Logger")

    handler = WarningHandler()
    parent = Logger.getLogger("group.gnometrading")
    parent.addHandler(handler)

    try:
        child = Logger.getLogger("group.gnometrading.backtest.driver.BacktestDriver")
        child.warning("test warning message")

        messages = list(handler.getMessages())
        assert len(messages) == 1
        assert str(messages[0]) == "test warning message"

        handler.clearMessages()
        assert len(list(handler.getMessages())) == 0
    finally:
        parent.removeHandler(handler)


@pytest.mark.skipif(not _jvm_available(), reason="JVM/gnome-backtest JARs not available")
def test_warning_handler_ignores_info_level():
    import jpype
    from gnomepy.java._jvm import ensure_jvm_started

    ensure_jvm_started()

    WarningHandler = jpype.JClass("group.gnometrading.backtest.recorder.WarningHandler")
    Logger = jpype.JClass("java.util.logging.Logger")

    handler = WarningHandler()
    parent = Logger.getLogger("group.gnometrading")
    parent.addHandler(handler)

    try:
        child = Logger.getLogger("group.gnometrading.backtest.driver.BacktestDriver")
        child.info("this is info, should not be captured")
        child.warning("this is a warning, should be captured")

        messages = list(handler.getMessages())
        assert len(messages) == 1
        assert str(messages[0]) == "this is a warning, should be captured"
    finally:
        parent.removeHandler(handler)
