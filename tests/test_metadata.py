"""Unit tests for BacktestMetadata and generate_backtest_id — no JVM needed."""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import pytest

from gnomepy.metadata import BacktestMetadata
from gnomepy.utils import generate_backtest_id


_ID_RE = re.compile(r"^[a-z][a-z0-9-]*-\d{8}-\d{6}-[0-9a-f]{4}$")


class TestGenerateBacktestId:
    def test_camel_class_name(self):
        uid = generate_backtest_id("MomentumTaker")
        assert uid.startswith("momentum-taker-")
        assert _ID_RE.match(uid)

    def test_python_import_path(self):
        uid = generate_backtest_id("gnomepy_research.strategies.momentum:MomentumTaker")
        assert uid.startswith("momentum-taker-")

    def test_java_fqn(self):
        uid = generate_backtest_id("com.example.strategies.MyStrategy")
        assert uid.startswith("my-strategy-")

    def test_no_strategy(self):
        uid = generate_backtest_id()
        assert uid.startswith("backtest-")
        assert _ID_RE.match(uid)

    def test_none_strategy(self):
        uid = generate_backtest_id(None)
        assert uid.startswith("backtest-")

    def test_market_maker(self):
        uid = generate_backtest_id("MarketMaker")
        assert uid.startswith("market-maker-")

    def test_unique(self):
        ids = {generate_backtest_id("TestStrategy") for _ in range(20)}
        # All should be unique (4 hex rand bytes → 65536 possibilities; 20 draws very unlikely to collide)
        assert len(ids) == 20


class TestBacktestMetadata:
    def test_requires_backtest_id(self):
        meta = BacktestMetadata(backtest_id="momentum-taker-20260415-103000-a3b2")
        assert meta.backtest_id == "momentum-taker-20260415-103000-a3b2"

    def test_created_at_auto_generated(self):
        meta = BacktestMetadata(backtest_id="test-id")
        assert meta.created_at is not None
        assert "T" in meta.created_at  # ISO format check

    def test_to_dict_omits_none(self):
        meta = BacktestMetadata(backtest_id="test-id", strategy=None, event_count=None)
        d = meta.to_dict()
        assert "strategy" not in d
        assert "event_count" not in d
        assert "backtest_id" in d
        assert "created_at" in d

    def test_to_dict_includes_set_fields(self):
        meta = BacktestMetadata(
            backtest_id="test-id",
            strategy="my.module:MyStrat",
            event_count=12345,
            wall_time_seconds=3.14,
        )
        d = meta.to_dict()
        assert d["strategy"] == "my.module:MyStrat"
        assert d["event_count"] == 12345
        assert d["wall_time_seconds"] == 3.14

    def test_roundtrip(self):
        meta = BacktestMetadata(
            backtest_id="momentum-taker-20260415-103000-a3b2",
            strategy="gnomepy_research.strategies.momentum:MomentumTaker",
            start_date="2026-01-23 10:30:00",
            end_date="2026-01-23 13:00:00",
            wall_time_seconds=42.3,
            event_count=1_234_567,
            preset_name="mm_btc_30m",
            gnomepy_version="1.0.0",
        )
        restored = BacktestMetadata.from_dict(meta.to_dict())
        assert restored.backtest_id == meta.backtest_id
        assert restored.strategy == meta.strategy
        assert restored.start_date == meta.start_date
        assert restored.event_count == meta.event_count
        assert restored.preset_name == meta.preset_name

    def test_from_dict_ignores_unknown_keys(self):
        d = {
            "backtest_id": "test-id",
            "created_at": "2026-01-01T00:00:00+00:00",
            "future_field": "some_value",
        }
        meta = BacktestMetadata.from_dict(d)
        assert meta.backtest_id == "test-id"
        assert meta.extra.get("future_field") == "some_value"

    def test_from_dict_missing_optional_fields(self):
        meta = BacktestMetadata.from_dict({"backtest_id": "test-id", "created_at": "2026-01-01T00:00:00+00:00"})
        assert meta.strategy is None
        assert meta.event_count is None
        assert meta.extra == {}

    def test_save_and_load(self):
        meta = BacktestMetadata(
            backtest_id="momentum-taker-20260415-103000-a3b2",
            strategy="my.module:Strat",
            event_count=999,
        )
        with tempfile.TemporaryDirectory() as tmp:
            meta.save(tmp)
            assert (Path(tmp) / "metadata.json").exists()
            loaded = BacktestMetadata.load(tmp)
        assert loaded is not None
        assert loaded.backtest_id == meta.backtest_id
        assert loaded.strategy == meta.strategy
        assert loaded.event_count == meta.event_count

    def test_load_returns_none_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = BacktestMetadata.load(tmp)
        assert result is None

    def test_saved_json_is_valid(self):
        meta = BacktestMetadata(backtest_id="test-id", strategy="foo:Bar")
        with tempfile.TemporaryDirectory() as tmp:
            meta.save(tmp)
            raw = (Path(tmp) / "metadata.json").read_text()
        parsed = json.loads(raw)
        assert parsed["backtest_id"] == "test-id"
        assert parsed["strategy"] == "foo:Bar"
        assert "strategy_args" not in parsed  # None field omitted
