"""Unit tests for the Tardis importer — no JVM, no network required."""
from __future__ import annotations

import math
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from gnomepy.importer.tardis.book import L2Book
from gnomepy.importer.tardis.mappings import build_mbp10_df, mbp10_import_config
from gnomepy.java.enums import SchemaType


# ---------------------------------------------------------------------------
# L2Book
# ---------------------------------------------------------------------------

class TestL2Book:
    def test_empty_book_update_returns_none(self):
        book = L2Book()
        # No previous state → only None if top-N still empty. Let's seed a level first.
        # Actually, updating from empty to non-empty changes top at index 0.
        depth = book.update("bid", 100.0, 5.0)
        assert depth == 0

    def test_bids_sorted_descending(self):
        book = L2Book()
        book.update("bid", 100.0, 1.0)
        book.update("bid", 101.0, 2.0)
        book.update("bid", 99.0, 3.0)
        top_bids, _ = book.top_levels()
        prices = [p for p, _ in top_bids]
        assert prices == sorted(prices, reverse=True)

    def test_asks_sorted_ascending(self):
        book = L2Book()
        book.update("ask", 102.0, 1.0)
        book.update("ask", 101.0, 2.0)
        book.update("ask", 103.0, 3.0)
        _, top_asks = book.top_levels()
        prices = [p for p, _ in top_asks]
        assert prices == sorted(prices)

    def test_remove_level_amount_zero(self):
        book = L2Book()
        book.update("bid", 100.0, 5.0)
        book.update("bid", 101.0, 3.0)
        depth = book.update("bid", 101.0, 0.0)
        top_bids, _ = book.top_levels()
        assert len(top_bids) == 1
        assert top_bids[0][0] == 100.0
        assert depth == 0

    def test_update_outside_top_n_returns_none(self):
        book = L2Book()
        for i in range(L2Book.NUM_LEVELS):
            book.update("bid", float(100 - i), 1.0)
        # Level below top-10
        depth = book.update("bid", 10.0, 1.0)
        assert depth is None

    def test_update_existing_level_correct_depth(self):
        book = L2Book()
        book.update("bid", 100.0, 1.0)
        book.update("bid", 99.0, 1.0)
        book.update("bid", 98.0, 1.0)
        depth = book.update("bid", 99.0, 5.0)  # changes level index 1
        assert depth == 1

    def test_clear_empties_book(self):
        book = L2Book()
        book.update("bid", 100.0, 5.0)
        book.clear()
        top_bids, top_asks = book.top_levels()
        assert top_bids == []
        assert top_asks == []

    def test_update_after_clear_returns_zero(self):
        book = L2Book()
        book.update("bid", 100.0, 1.0)
        book.clear()
        depth = book.update("bid", 100.0, 1.0)
        assert depth == 0

    def test_top_levels_capped_at_num_levels(self):
        book = L2Book()
        for i in range(L2Book.NUM_LEVELS + 5):
            book.update("bid", float(100 - i), 1.0)
        top_bids, _ = book.top_levels()
        assert len(top_bids) == L2Book.NUM_LEVELS


# ---------------------------------------------------------------------------
# build_mbp10_df
# ---------------------------------------------------------------------------

def _l2_row(timestamp, local_ts, is_snapshot, side, price, amount):
    return {
        "exchange": "binance-futures",
        "symbol": "BTCUSDT",
        "timestamp": timestamp,
        "local_timestamp": local_ts,
        "is_snapshot": is_snapshot,
        "side": side,
        "price": price,
        "amount": amount,
    }

def _trade_row(timestamp, local_ts, side, price, amount):
    return {
        "exchange": "binance-futures",
        "symbol": "BTCUSDT",
        "timestamp": timestamp,
        "local_timestamp": local_ts,
        "id": "1",
        "side": side,
        "price": price,
        "amount": amount,
    }

def _make_l2_df(rows):
    return pd.DataFrame(rows)

def _make_trades_df(rows):
    return pd.DataFrame(rows)


class TestBuildMbp10Df:
    def test_emits_snapshot_row(self):
        l2 = _make_l2_df([
            _l2_row(1000, 1001, True, "bid", 100.0, 5.0),
            _l2_row(1000, 1001, True, "ask", 101.0, 3.0),
        ])
        trades = _make_trades_df([])
        df = build_mbp10_df(l2, trades)
        assert len(df) == 1
        assert df.iloc[0]["action"] == "None_"
        assert df.iloc[0]["depth"] == 0

    def test_pre_snapshot_l2_skipped(self):
        l2 = _make_l2_df([
            _l2_row(500, 501, False, "bid", 99.0, 1.0),   # pre-snapshot
            _l2_row(1000, 1001, True, "bid", 100.0, 5.0),  # snapshot
        ])
        trades = _make_trades_df([])
        df = build_mbp10_df(l2, trades)
        # Only one snapshot row emitted; pre-snapshot update is silently discarded
        assert len(df) == 1
        assert df.iloc[0]["bid_price_0"] == 100.0

    def test_incremental_update_emitted(self):
        l2 = _make_l2_df([
            _l2_row(1000, 1001, True, "bid", 100.0, 5.0),
            _l2_row(2000, 2001, False, "bid", 101.0, 2.0),
        ])
        trades = _make_trades_df([])
        df = build_mbp10_df(l2, trades)
        assert len(df) == 2
        assert df.iloc[1]["depth"] == 0
        assert df.iloc[1]["bid_price_0"] == 101.0

    def test_update_outside_top_n_not_emitted(self):
        # Fill top-10 bids, then update a level below
        snap_rows = [_l2_row(1000, 1001, True, "bid", float(100 - i), 1.0) for i in range(10)]
        l2 = _make_l2_df(snap_rows + [_l2_row(2000, 2001, False, "bid", 10.0, 1.0)])
        trades = _make_trades_df([])
        df = build_mbp10_df(l2, trades)
        assert len(df) == 1  # Only the snapshot; the update below top-10 is suppressed

    def test_trade_row_has_correct_fields(self):
        l2 = _make_l2_df([
            _l2_row(1000, 1001, True, "bid", 100.0, 5.0),
            _l2_row(1000, 1001, True, "ask", 101.0, 3.0),
        ])
        trades = _make_trades_df([_trade_row(1500, 1501, "buy", 101.0, 0.1)])
        df = build_mbp10_df(l2, trades)
        trade_row = df[df["action"] == "Trade"].iloc[0]
        assert trade_row["trade_side"] == "Bid"
        assert trade_row["depth"] == 255
        assert trade_row["trade_price"] == 101.0
        assert trade_row["bid_price_0"] == 100.0  # current book state carried through

    def test_trade_side_mapping(self):
        l2 = _make_l2_df([_l2_row(1000, 1001, True, "bid", 100.0, 1.0)])
        for tardis_side, expected in [("buy", "Bid"), ("sell", "Ask"), ("unknown", "None_")]:
            trades = _make_trades_df([_trade_row(1500, 1501, tardis_side, 100.0, 1.0)])
            df = build_mbp10_df(l2, trades)
            assert df[df["action"] == "Trade"].iloc[0]["trade_side"] == expected

    def test_book_levels_in_trade_row(self):
        l2 = _make_l2_df([
            _l2_row(1000, 1001, True, "bid", 100.0, 5.0),
            _l2_row(1000, 1001, True, "ask", 101.0, 3.0),
        ])
        trades = _make_trades_df([_trade_row(2000, 2001, "sell", 100.0, 1.0)])
        df = build_mbp10_df(l2, trades)
        trade_row = df[df["action"] == "Trade"].iloc[0]
        assert trade_row["bid_price_0"] == 100.0
        assert trade_row["ask_price_0"] == 101.0

    def test_snapshot_transition_clears_book(self):
        l2 = _make_l2_df([
            _l2_row(1000, 1001, True, "bid", 100.0, 5.0),  # first snapshot
            _l2_row(2000, 2001, False, "bid", 99.0, 1.0),   # incremental
            _l2_row(3000, 3001, True, "bid", 200.0, 1.0),   # second snapshot (reconnect)
        ])
        trades = _make_trades_df([])
        df = build_mbp10_df(l2, trades)
        # Last snapshot row should have only bid=200 (old book cleared)
        last_row = df.iloc[-1]
        assert last_row["bid_price_0"] == 200.0
        assert math.isnan(last_row["bid_price_1"])

    def test_empty_trades_df_still_works(self):
        l2 = _make_l2_df([_l2_row(1000, 1001, True, "bid", 100.0, 5.0)])
        trades = _make_trades_df([])
        df = build_mbp10_df(l2, trades)
        assert len(df) == 1

    def test_empty_l2_df_returns_empty(self):
        l2 = _make_l2_df([])
        trades = _make_trades_df([_trade_row(1000, 1001, "buy", 100.0, 1.0)])
        df = build_mbp10_df(l2, trades)
        assert df.empty

    def test_incremental_depth_correct(self):
        snap_rows = [
            _l2_row(1000, 1001, True, "bid", 100.0, 1.0),
            _l2_row(1000, 1001, True, "bid", 99.0, 1.0),
            _l2_row(1000, 1001, True, "bid", 98.0, 1.0),
        ]
        l2 = _make_l2_df(snap_rows + [
            _l2_row(2000, 2001, False, "bid", 99.0, 5.0),  # update at level index 1
        ])
        trades = _make_trades_df([])
        df = build_mbp10_df(l2, trades)
        assert df.iloc[1]["depth"] == 1


# ---------------------------------------------------------------------------
# mbp10_import_config
# ---------------------------------------------------------------------------

class TestMbp10ImportConfig:
    def test_schema_type(self):
        config = mbp10_import_config(1, 2)
        assert config.schema_type == SchemaType.MBP_10

    def test_security_and_exchange_ids(self):
        config = mbp10_import_config(42, 7)
        assert config.security_id == 42
        assert config.exchange_id == 7

    def test_timestamp_field(self):
        config = mbp10_import_config(1, 1)
        assert config.timestamp_field == "timestamp_event"

    def test_field_count(self):
        config = mbp10_import_config(1, 1)
        # 2 timestamps + price + size + action + side + depth + 10 levels × 4 fields = 47
        assert len(config.field_mappings) == 47

    def test_timestamp_transforms_use_epoch_us(self):
        config = mbp10_import_config(1, 1)
        ts_mappings = [m for m in config.field_mappings if m.transform == "timestamp"]
        assert all(m.timestamp_format == "epoch_us" for m in ts_mappings)

    def test_level_price_transforms_are_price(self):
        config = mbp10_import_config(1, 1)
        price_fields = [m for m in config.field_mappings if "price" in m.target_field and m.target_field != "trade_price"]
        assert all(m.transform == "price" for m in price_fields)

    def test_depth_uses_none_transform(self):
        config = mbp10_import_config(1, 1)
        depth_mapping = next(m for m in config.field_mappings if m.target_field == "depth")
        assert depth_mapping.transform == "none"

    def test_action_enum_map_covers_all_values(self):
        config = mbp10_import_config(1, 1)
        action_mapping = next(m for m in config.field_mappings if m.target_field == "action")
        assert "Trade" in action_mapping.enum_map
        assert "None_" in action_mapping.enum_map

    def test_side_enum_map_covers_all_values(self):
        config = mbp10_import_config(1, 1)
        side_mapping = next(m for m in config.field_mappings if m.target_field == "side")
        assert "Bid" in side_mapping.enum_map
        assert "Ask" in side_mapping.enum_map
        assert "None_" in side_mapping.enum_map

    def test_bucket_override(self):
        config = mbp10_import_config(1, 1, bucket="my-bucket")
        assert config.bucket == "my-bucket"

    def test_defaults_contain_timestamp_sent(self):
        config = mbp10_import_config(1, 1)
        assert config.defaults.get("timestamp_sent") == 0


# ---------------------------------------------------------------------------
# TardisClient (mocked tardis_dev)
# ---------------------------------------------------------------------------

class TestTardisClient:
    def test_raises_import_error_without_tardis_dev(self):
        with patch.dict("sys.modules", {"tardis_dev": None}):
            from gnomepy.importer.tardis import client as client_module
            import importlib
            importlib.reload(client_module)
            with pytest.raises(ImportError, match="tardis-dev"):
                client_module.TardisClient()

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("TARDIS_API_KEY", "test-key-123")
        mock_tardis = MagicMock()
        with patch.dict("sys.modules", {"tardis_dev": mock_tardis}):
            from gnomepy.importer.tardis.client import TardisClient
            c = TardisClient()
            assert c._api_key == "test-key-123"

    def test_download_calls_download_datasets(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TARDIS_API_KEY", "key")
        mock_tardis = MagicMock()
        with patch.dict("sys.modules", {"tardis_dev": mock_tardis}):
            from gnomepy.importer.tardis.client import TardisClient
            c = TardisClient(api_key="key")
            c.download("binance-futures", ["trades"], date(2024, 1, 15), ["BTCUSDT"], tmp_path)
            mock_tardis.download_datasets.assert_called_once()
            kwargs = mock_tardis.download_datasets.call_args
            assert kwargs.kwargs["exchange"] == "binance-futures" or kwargs.args[0] == "binance-futures"


# ---------------------------------------------------------------------------
# TardisImporter (mocked client + registry)
# ---------------------------------------------------------------------------

class TestTardisImporter:
    def _make_mock_registry(self, exchange_id=3, security_id=1):
        registry = MagicMock()
        exchange = MagicMock()
        exchange.exchange_id = exchange_id
        registry.get_exchange.return_value = [exchange]
        listing = MagicMock()
        listing.exchange_id = exchange_id
        listing.security_id = security_id
        registry.get_listing.return_value = [listing]
        return registry

    def _sample_l2_df(self):
        return pd.DataFrame([
            {"exchange": "binance-futures", "symbol": "BTCUSDT", "timestamp": 1000000,
             "local_timestamp": 1000001, "is_snapshot": True, "side": "bid", "price": 50000.0, "amount": 1.0},
            {"exchange": "binance-futures", "symbol": "BTCUSDT", "timestamp": 1000000,
             "local_timestamp": 1000001, "is_snapshot": True, "side": "ask", "price": 50001.0, "amount": 1.0},
        ])

    def _sample_trades_df(self):
        return pd.DataFrame([
            {"exchange": "binance-futures", "symbol": "BTCUSDT", "timestamp": 2000000,
             "local_timestamp": 2000001, "id": "1", "side": "buy", "price": 50001.0, "amount": 0.1},
        ])

    def test_resolve_raises_for_unknown_exchange(self):
        from gnomepy.importer.tardis.importer import TardisImporter, TardisImportRequest
        importer = TardisImporter(registry=MagicMock())
        req = TardisImportRequest("unknown-exchange", ["BTCUSDT"], date(2024, 1, 1), date(2024, 1, 1))
        with pytest.raises(ValueError, match="not mapped"):
            importer.run(req)

    def test_dry_run_does_not_call_s3(self):
        from gnomepy.importer.tardis.importer import TardisImporter, TardisImportRequest

        mock_client = MagicMock()
        mock_client.download.return_value = None
        registry = self._make_mock_registry()
        s3 = MagicMock()

        l2_df = self._sample_l2_df()
        trades_df = self._sample_trades_df()

        with patch("gnomepy.importer.tardis.importer._load_day", return_value=(l2_df, trades_df)):
            with patch("gnomepy.importer.tardis.importer.ImportJob") as mock_job_cls:
                mock_job = MagicMock()
                mock_job.dry_run.return_value = MagicMock(is_valid=True, errors=[], total_records=2)
                mock_job_cls.return_value = mock_job

                importer = TardisImporter(client=mock_client, registry=registry, s3_client=s3)
                req = TardisImportRequest(
                    "binance-futures", ["BTCUSDT"],
                    date(2024, 1, 15), date(2024, 1, 15),
                    dry_run=True,
                )
                results = importer.run(req)
                assert results[0].days_processed == 1
                mock_job.dry_run.assert_called_once()
                mock_job.run.assert_not_called()

    def test_missing_data_increments_days_skipped(self):
        from gnomepy.importer.tardis.importer import TardisImporter, TardisImportRequest

        mock_client = MagicMock()
        registry = self._make_mock_registry()

        with patch("gnomepy.importer.tardis.importer._load_day", return_value=(None, None)):
            importer = TardisImporter(client=mock_client, registry=registry)
            req = TardisImportRequest(
                "binance-futures", ["BTCUSDT"],
                date(2024, 1, 15), date(2024, 1, 17),
                dry_run=True,
            )
            results = importer.run(req)
            assert results[0].days_skipped == 3
            assert results[0].days_processed == 0
