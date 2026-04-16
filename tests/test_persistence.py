"""Round-trip and format tests for BacktestResults persistence — no JVM needed."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from gnomepy.java.recorder import BacktestResults
from gnomepy.metadata import BacktestMetadata


def _make_market_df() -> pd.DataFrame:
    ts = pd.to_datetime(["2026-01-23 10:30:00", "2026-01-23 10:30:01"], utc=True)
    return pd.DataFrame(
        {
            "exchange_id": [1, 1],
            "security_id": [101, 101],
            "bid_price_0": [50000.0, 50001.0],
            "ask_price_0": [50001.0, 50002.0],
            "bid_size_0": [1.0, 0.5],
            "ask_size_0": [0.5, 1.0],
            "last_trade_price": [50000.5, 50001.5],
            "last_trade_size": [0.1, 0.2],
        },
        index=pd.DatetimeIndex(ts, name="timestamp"),
    )


def _make_fills_df() -> pd.DataFrame:
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


def _make_metadata() -> BacktestMetadata:
    return BacktestMetadata(
        backtest_id="market-maker-20260123-103000-ab12",
        strategy="gnomepy_research.strategies:MarketMaker",
        start_date="2026-01-23 10:30:00",
        end_date="2026-01-23 13:00:00",
        wall_time_seconds=5.3,
        event_count=1234,
    )


class TestFromDataframes:
    def test_empty(self):
        r = BacktestResults.from_dataframes()
        assert r.market_records_df().empty
        assert r.fills_df().empty
        assert r.custom_metrics() == {}
        assert r.backtest_id is None

    def test_with_data(self):
        market = _make_market_df()
        r = BacktestResults.from_dataframes(market_df=market)
        assert len(r.market_records_df()) == 2
        assert r.fills_df().empty

    def test_with_metadata(self):
        meta = _make_metadata()
        r = BacktestResults.from_dataframes(metadata=meta)
        assert r.backtest_id == "market-maker-20260123-103000-ab12"
        assert r.metadata.strategy == "gnomepy_research.strategies:MarketMaker"

    def test_with_custom_metrics(self):
        signals = pd.DataFrame({"value": [1.0, 2.0]})
        r = BacktestResults.from_dataframes(custom_metrics={"signals": signals})
        result = r.custom_metrics("signals")
        assert len(result) == 2
        assert list(result["value"]) == [1.0, 2.0]

    def test_record_counts(self):
        market = _make_market_df()
        fills = _make_fills_df()
        r = BacktestResults.from_dataframes(market_df=market, fills_df=fills)
        assert r.market_record_count == 2
        assert r.fill_record_count == 1
        assert r.order_record_count == 0
        assert r.intent_record_count == 0


class TestSaveAndLoad:
    def test_round_trip_market(self):
        market = _make_market_df()
        r = BacktestResults.from_dataframes(market_df=market)
        with tempfile.TemporaryDirectory() as tmp:
            r.save(tmp)
            loaded = BacktestResults.from_parquet(tmp)
        pd.testing.assert_frame_equal(market, loaded.market_records_df())

    def test_round_trip_fills(self):
        fills = _make_fills_df()
        r = BacktestResults.from_dataframes(fills_df=fills)
        with tempfile.TemporaryDirectory() as tmp:
            r.save(tmp)
            loaded = BacktestResults.from_parquet(tmp)
        pd.testing.assert_frame_equal(fills, loaded.fills_df())

    def test_round_trip_metadata(self):
        meta = _make_metadata()
        r = BacktestResults.from_dataframes(metadata=meta)
        with tempfile.TemporaryDirectory() as tmp:
            r.save(tmp)
            loaded = BacktestResults.from_parquet(tmp)
        assert loaded.backtest_id == meta.backtest_id
        assert loaded.metadata.strategy == meta.strategy
        assert loaded.metadata.event_count == meta.event_count
        assert loaded.metadata.data_scaled is True

    def test_metadata_json_has_data_scaled(self):
        meta = _make_metadata()
        r = BacktestResults.from_dataframes(metadata=meta)
        with tempfile.TemporaryDirectory() as tmp:
            r.save(tmp)
            raw = json.loads((Path(tmp) / "metadata.json").read_text())
        assert raw["data_scaled"] is True

    def test_file_layout(self):
        market = _make_market_df()
        fills = _make_fills_df()
        r = BacktestResults.from_dataframes(market_df=market, fills_df=fills, metadata=_make_metadata())
        with tempfile.TemporaryDirectory() as tmp:
            r.save(tmp)
            files = {f.name for f in Path(tmp).iterdir()}
        assert "market.parquet" in files
        assert "fills.parquet" in files
        assert "metadata.json" in files
        # Empty streams are not written
        assert "orders.parquet" not in files
        assert "intents.parquet" not in files

    def test_round_trip_custom_metrics(self):
        signals = pd.DataFrame({"value": [1.0, 2.0], "ts": [100, 200]})
        r = BacktestResults.from_dataframes(custom_metrics={"signals": signals})
        with tempfile.TemporaryDirectory() as tmp:
            r.save(tmp)
            # Custom dir should exist
            assert (Path(tmp) / "custom" / "signals.parquet").exists()
            loaded = BacktestResults.from_parquet(tmp)
        result = loaded.custom_metrics("signals")
        assert list(result["value"]) == [1.0, 2.0]

    def test_all_empty(self):
        r = BacktestResults.from_dataframes()
        with tempfile.TemporaryDirectory() as tmp:
            r.save(tmp)
            loaded = BacktestResults.from_parquet(tmp)
        assert loaded.market_records_df().empty
        assert loaded.fills_df().empty
        assert loaded.custom_metrics() == {}

    def test_load_missing_metadata(self):
        """from_parquet returns None metadata when no metadata.json is present."""
        # save() skips metadata.json when _metadata is None
        r = BacktestResults.from_dataframes(market_df=_make_market_df())
        with tempfile.TemporaryDirectory() as tmp:
            r.save(tmp)
            assert not (Path(tmp) / "metadata.json").exists()
            loaded = BacktestResults.from_parquet(tmp)
        assert loaded.metadata is None
        assert len(loaded.market_records_df()) == 2

    def test_save_uses_path_object(self):
        r = BacktestResults.from_dataframes(market_df=_make_market_df())
        with tempfile.TemporaryDirectory() as tmp:
            r.save(Path(tmp))  # Path object, not str
            loaded = BacktestResults.from_parquet(Path(tmp))
        assert len(loaded.market_records_df()) == 2

    def test_repr_from_parquet(self):
        r = BacktestResults.from_dataframes(metadata=_make_metadata())
        with tempfile.TemporaryDirectory() as tmp:
            r.save(tmp)
            loaded = BacktestResults.from_parquet(tmp)
        assert "from_parquet=True" in repr(loaded)
        assert "market-maker-20260123-103000-ab12" in repr(loaded)


class TestCustomMetrics:
    def test_custom_metrics_dict(self):
        signals = pd.DataFrame({"a": [1, 2]})
        features = pd.DataFrame({"b": [3, 4]})
        r = BacktestResults.from_dataframes(
            custom_metrics={"signals": signals, "features": features}
        )
        m = r.custom_metrics()
        assert set(m.keys()) == {"signals", "features"}

    def test_custom_metrics_missing_name(self):
        r = BacktestResults.from_dataframes()
        assert r.custom_metrics("nonexistent").empty

    def test_custom_metrics_round_trip_multiple(self):
        r = BacktestResults.from_dataframes(
            custom_metrics={
                "signals": pd.DataFrame({"x": [1.0]}),
                "features": pd.DataFrame({"y": [2.0]}),
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            r.save(tmp)
            loaded = BacktestResults.from_parquet(tmp)
        assert set(loaded.custom_metrics().keys()) == {"signals", "features"}
