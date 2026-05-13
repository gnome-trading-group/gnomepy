"""Integration test: verify S3 missing-key warnings reach BacktestResults.metadata.warnings.

Listing ID 1 on 2026-01-04 has missing minutes in S3.
Requires: JVM + gnome-backtest JARs (GNOME_JARS) + AWS credentials.
"""
from __future__ import annotations

from datetime import date

import pytest


def _jvm_available() -> bool:
    try:
        from gnomepy.java._classpath import discover_classpath
        return len(discover_classpath()) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _jvm_available(), reason="JVM/gnome-backtest JARs not available"
)


def _make_noop_strategy():
    from gnomepy.java.backtest.strategy import Strategy

    class NoOpStrategy(Strategy):
        def on_market_data(self, data):
            return []

        def on_execution_report(self, report):
            return []

    return NoOpStrategy()


def test_s3_missing_key_warning_captured():
    """Listing 1 on 2026-01-04 has missing S3 minutes.
    At least one warning must appear in results.metadata.warnings and in summary()['warnings'].
    """
    from gnomepy.java.backtest.config import (
        BacktestConfig,
        ExchangeProfileConfig,
        ListingSimConfig,
    )
    from gnomepy.java.backtest.runner import Backtest
    from gnomepy.reporting.report import BacktestReport

    config = BacktestConfig(
        start_date=date(2026, 1, 4),
        end_date=date(2026, 1, 5),
        listings=[ListingSimConfig(listing_id=1, profile="default")],
        profiles={"default": ExchangeProfileConfig()},
    )

    bt = Backtest(config, strategy=_make_noop_strategy())
    results = bt.run(progress=False)

    assert results is not None, "Backtest returned None — recording may be disabled"
    assert results.metadata is not None, "BacktestResults has no metadata"

    warnings = results.metadata.warnings
    s3_warnings = [w for w in warnings if "Missing S3 key" in w]

    assert len(s3_warnings) >= 1, (
        f"Expected at least 1 missing-key warning for listing 1 on 2026-01-04, "
        f"got {len(s3_warnings)}: {s3_warnings!r}\n"
        f"All warnings: {warnings!r}"
    )

    # Also verify the warning flows through the report summary
    report = BacktestReport(results)
    summary_warnings = report.summary()["warnings"]
    assert summary_warnings == warnings, (
        f"summary()['warnings'] != results.metadata.warnings\n"
        f"  summary: {summary_warnings!r}\n"
        f"  metadata: {warnings!r}"
    )
