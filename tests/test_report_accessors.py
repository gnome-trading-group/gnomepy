"""Public frame accessors on BacktestReport — no JVM needed.

Callers previously had to reach into ``report._market_df`` / ``report._intent_df``
to feed the analysis helpers, because the enriched market frame (the one carrying
``mid_price``) is built inside ``BacktestReport`` and was not exposed.
"""
from __future__ import annotations

import pandas as pd
import pytest

from gnomepy.java.recorder import BacktestResults
from gnomepy.metadata import BacktestMetadata
from gnomepy.reporting.report import BacktestReport


def _market_df() -> pd.DataFrame:
    ts = pd.to_datetime(["2026-01-23 10:30:00", "2026-01-23 10:30:01"], utc=True)
    return pd.DataFrame(
        {
            "exchange_id": [1, 1],
            "security_id": [101, 101],
            "bid_price_0": [50000.0, 50001.0],
            "ask_price_0": [50001.0, 50002.0],
            "bid_size_0": [1.0, 0.5],
            "ask_size_0": [0.5, 1.0],
        },
        index=pd.DatetimeIndex(ts, name="timestamp"),
    )


def _fills_df() -> pd.DataFrame:
    ts = pd.to_datetime(["2026-01-23 10:30:00"], utc=True)
    return pd.DataFrame(
        {
            "exchange_id": [1],
            "security_id": [101],
            "strategy_id": [0],
            "client_oid": [1001],
            "side": ["Bid"],
            "fill_price": [50000.5],
            "fill_qty": [0.1],
            "leaves_qty": [0.0],
            "fee": [0.001],
            "book_bid_price": [50000.0],
            "book_ask_price": [50001.0],
        },
        index=pd.DatetimeIndex(ts, name="timestamp"),
    )


@pytest.fixture
def report() -> BacktestReport:
    results = BacktestResults.from_dataframes(
        market_df=_market_df(),
        fills_df=_fills_df(),
        metadata=BacktestMetadata(backtest_id="test-id"),
    )
    return BacktestReport(results)


@pytest.fixture
def frame_report() -> BacktestReport:
    """A report built via from_dataframes, which leaves ``_results`` as None."""
    return BacktestReport.from_dataframes(_market_df(), _fills_df())


class TestMarketDf:
    def test_is_public(self, report):
        assert isinstance(report.market_df, pd.DataFrame)
        assert not report.market_df.empty

    def test_carries_mid_price(self, report):
        """The whole reason the accessor is needed: analysis helpers require this column."""
        assert "mid_price" in report.market_df.columns
        assert report.market_df["mid_price"].iloc[0] == pytest.approx(50000.5)

    def test_available_on_both_construction_paths(self, report, frame_report):
        assert "mid_price" in frame_report.market_df.columns
        assert len(frame_report.market_df) == len(report.market_df)


class TestIntentDf:
    def test_is_public(self, report):
        assert isinstance(report.intent_df, pd.DataFrame)

    def test_available_on_both_construction_paths(self, frame_report):
        assert isinstance(frame_report.intent_df, pd.DataFrame)


class TestCustomMetrics:
    def test_returns_dict_when_unnamed(self, report):
        assert isinstance(report.custom_metrics(), dict)

    def test_returns_frame_when_named(self, report):
        assert isinstance(report.custom_metrics("nope"), pd.DataFrame)

    def test_frame_built_report_returns_empty_rather_than_raising(self, frame_report):
        """from_dataframes leaves _results as None; the passthrough must not explode."""
        assert frame_report.custom_metrics() == {}
        assert frame_report.custom_metrics("anything").empty


class TestMetadata:
    def test_returns_metadata_when_present(self, report):
        assert report.metadata.backtest_id == "test-id"

    def test_frame_built_report_returns_none_rather_than_raising(self, frame_report):
        """Documented as 'None if not available', but the guard was missing."""
        assert frame_report.metadata is None
