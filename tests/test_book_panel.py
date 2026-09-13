"""Tests for book panel data methods and ladder builder."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from dash import html

from gnomepy.explorer.panels.book import (
    _merge_with_intent,
    _near_fill,
    build_ladder,
)


def _market_df() -> pd.DataFrame:
    ts = pd.to_datetime([
        "2026-01-23 10:30:00",
        "2026-01-23 10:30:01",
        "2026-01-23 10:30:02",
        "2026-01-23 10:30:03",
    ], utc=True)
    return pd.DataFrame(
        {
            "exchange_id": [1, 2, 1, 2],
            "security_id": [101, 201, 101, 201],
            "mid_price": [0.60, 0.59, 0.61, 0.60],
            "bid_price_0": [0.59, 0.58, 0.60, 0.59],
            "bid_size_0": [100.0, 80.0, 110.0, 90.0],
            "bid_price_1": [0.58, 0.57, 0.59, 0.58],
            "bid_size_1": [200.0, 150.0, 220.0, 160.0],
            "ask_price_0": [0.61, 0.60, 0.62, 0.61],
            "ask_size_0": [120.0, 90.0, 130.0, 100.0],
            "ask_price_1": [0.62, 0.61, 0.63, 0.62],
            "ask_size_1": [250.0, 180.0, 260.0, 190.0],
        },
        index=pd.DatetimeIndex(ts, name="timestamp"),
    )


def _fills_df() -> pd.DataFrame:
    ts = pd.to_datetime(["2026-01-23 10:30:00.200"], utc=True)
    return pd.DataFrame(
        {
            "exchange_id": [1],
            "security_id": [101],
            "side": ["Bid"],
            "fill_price": [0.59],
            "fill_qty": [50.0],
            "fee": [0.001],
            "book_bid_price": [0.59],
            "book_ask_price": [0.61],
        },
        index=pd.DatetimeIndex(ts, name="timestamp"),
    )


def _intents_df() -> pd.DataFrame:
    ts = pd.to_datetime(["2026-01-23 10:30:00.100", "2026-01-23 10:30:02.100"], utc=True)
    return pd.DataFrame(
        {
            "exchange_id": [1, 1],
            "security_id": [101, 101],
            "bid_price": [0.595, 0.605],
            "bid_size": [30.0, 35.0],
            "ask_price": [0.615, 0.625],
            "ask_size": [30.0, 35.0],
            "take_limit_price": [0.0, 0.0],
            "take_side": ["", ""],
            "take_size": [0.0, 0.0],
        },
        index=pd.DatetimeIndex(ts, name="timestamp"),
    )


def _make_store(market_df, fills_df, intents_df):
    from gnomepy.explorer.data import ExplorerDataStore, _detect_record_depth
    store = object.__new__(ExplorerDataStore)
    store.label = "A"
    store.metadata = None
    store.market_df = market_df
    store.fills_df = fills_df
    store.orders_df = pd.DataFrame()
    store.intents_df = intents_df
    store.custom_dfs = {}
    store.record_depth = _detect_record_depth(market_df)
    store.t_min = market_df.index[0]
    store.t_max = market_df.index[-1]
    store._event_ts_by_type = {}
    store._listing_labels = {}
    return store


class TestBookSnapshot:
    def test_returns_correct_listing(self):
        store = _make_store(_market_df(), _fills_df(), _intents_df())
        ts = pd.Timestamp("2026-01-23 10:30:01", tz="UTC")
        result = store.book_snapshot(ts, listing=(1, 101))
        assert (1, 101) in result
        data = result[(1, 101)]
        assert data["bids"][0][0] == pytest.approx(0.59)
        assert data["asks"][0][0] == pytest.approx(0.61)

    def test_finds_latest_row_before_timestamp(self):
        store = _make_store(_market_df(), _fills_df(), _intents_df())
        ts = pd.Timestamp("2026-01-23 10:30:01.500", tz="UTC")
        result = store.book_snapshot(ts, listing=(1, 101))
        data = result[(1, 101)]
        assert data["bids"][0][0] == pytest.approx(0.59)

    def test_returns_all_listings_when_no_filter(self):
        store = _make_store(_market_df(), _fills_df(), _intents_df())
        ts = pd.Timestamp("2026-01-23 10:30:02", tz="UTC")
        result = store.book_snapshot(ts)
        assert (1, 101) in result
        assert (2, 201) in result

    def test_before_first_row_returns_empty(self):
        store = _make_store(_market_df(), _fills_df(), _intents_df())
        ts = pd.Timestamp("2026-01-23 09:00:00", tz="UTC")
        result = store.book_snapshot(ts, listing=(1, 101))
        assert (1, 101) not in result

    def test_depth_levels_populated(self):
        store = _make_store(_market_df(), _fills_df(), _intents_df())
        ts = pd.Timestamp("2026-01-23 10:30:03", tz="UTC")
        result = store.book_snapshot(ts, listing=(1, 101))
        data = result[(1, 101)]
        assert len(data["bids"]) == 2
        assert len(data["asks"]) == 2


class TestIntentAt:
    def test_returns_most_recent_intent(self):
        store = _make_store(_market_df(), _fills_df(), _intents_df())
        ts = pd.Timestamp("2026-01-23 10:30:01", tz="UTC")
        result = store.intent_at(ts, listing=(1, 101))
        assert (1, 101) in result
        assert result[(1, 101)]["bid_price"] == pytest.approx(0.595)

    def test_second_intent_at_later_time(self):
        store = _make_store(_market_df(), _fills_df(), _intents_df())
        ts = pd.Timestamp("2026-01-23 10:30:03", tz="UTC")
        result = store.intent_at(ts, listing=(1, 101))
        assert result[(1, 101)]["bid_price"] == pytest.approx(0.605)

    def test_returns_empty_before_first_intent(self):
        store = _make_store(_market_df(), _fills_df(), _intents_df())
        ts = pd.Timestamp("2026-01-23 10:30:00.050", tz="UTC")
        result = store.intent_at(ts, listing=(1, 101))
        assert (1, 101) not in result

    def test_empty_when_no_intents(self):
        store = _make_store(_market_df(), _fills_df(), pd.DataFrame())
        ts = pd.Timestamp("2026-01-23 10:30:01", tz="UTC")
        result = store.intent_at(ts)
        assert result == {}


class TestFillsNear:
    def test_returns_fill_within_window(self):
        store = _make_store(_market_df(), _fills_df(), _intents_df())
        ts = pd.Timestamp("2026-01-23 10:30:00", tz="UTC")
        result = store.fills_near(ts, window_ns=500_000_000)
        assert len(result) == 1
        assert float(result.iloc[0]["fill_price"]) == pytest.approx(0.59)

    def test_excludes_fill_outside_window(self):
        store = _make_store(_market_df(), _fills_df(), _intents_df())
        ts = pd.Timestamp("2026-01-23 10:30:03", tz="UTC")
        result = store.fills_near(ts, window_ns=500_000_000)
        assert len(result) == 0

    def test_listing_filter(self):
        store = _make_store(_market_df(), _fills_df(), _intents_df())
        ts = pd.Timestamp("2026-01-23 10:30:00", tz="UTC")
        result = store.fills_near(ts, listing=(2, 201))
        assert len(result) == 0

    def test_empty_fills_returns_empty(self):
        store = _make_store(_market_df(), pd.DataFrame(), _intents_df())
        ts = pd.Timestamp("2026-01-23 10:30:00", tz="UTC")
        result = store.fills_near(ts)
        assert result.empty


class TestMergeWithIntent:
    def test_no_intent(self):
        levels = [(0.61, 100.0), (0.62, 200.0)]
        result = _merge_with_intent(levels, 0.0, 0.0)
        assert all(not is_intent for _, _, is_intent in result)
        assert len(result) == 2

    def test_intent_matches_level(self):
        levels = [(0.61, 100.0), (0.62, 200.0)]
        result = _merge_with_intent(levels, 0.61, 50.0)
        intent_rows = [(p, s, f) for p, s, f in result if f]
        assert len(intent_rows) == 1
        assert intent_rows[0][0] == pytest.approx(0.61)

    def test_intent_inserted_between_levels(self):
        levels = [(0.62, 100.0), (0.60, 200.0)]
        result = _merge_with_intent(levels, 0.61, 50.0)
        prices = [p for p, _, _ in result]
        assert prices == pytest.approx([0.62, 0.61, 0.60])
        assert result[1][2] is True

    def test_intent_inserted_at_top(self):
        levels = [(0.61, 100.0), (0.60, 200.0)]
        result = _merge_with_intent(levels, 0.63, 50.0)
        assert result[0] == pytest.approx((0.63, 50.0, True), rel=1e-5)

    def test_intent_appended_at_bottom(self):
        levels = [(0.62, 100.0), (0.61, 200.0)]
        result = _merge_with_intent(levels, 0.59, 50.0)
        assert result[-1][0] == pytest.approx(0.59)
        assert result[-1][2] is True


class TestNearFill:
    def test_finds_fill_at_price(self):
        fills = _fills_df()
        found, qty = _near_fill(0.59, fills)
        assert found is True
        assert qty == pytest.approx(50.0)

    def test_no_fill_at_price(self):
        fills = _fills_df()
        found, qty = _near_fill(0.61, fills)
        assert found is False

    def test_empty_fills(self):
        found, qty = _near_fill(0.59, pd.DataFrame())
        assert found is False


class TestBuildLadder:
    def test_returns_div(self):
        book_data = {
            "bids": [(0.59, 100.0), (0.58, 200.0)],
            "asks": [(0.61, 120.0), (0.62, 250.0)],
            "timestamp": pd.Timestamp("2026-01-23 10:30:00", tz="UTC"),
        }
        result = build_ladder(book_data, None, pd.DataFrame(), "TEST @ Venue", 2, 2)
        assert isinstance(result, html.Div)

    def test_no_book_data_shows_placeholder(self):
        result = build_ladder(None, None, pd.DataFrame(), "TEST", 2, 1)
        assert isinstance(result, html.Div)
        assert "No book data" in str(result)

    def test_intent_not_in_book_inserts_row(self):
        book_data = {
            "bids": [(0.59, 100.0)],
            "asks": [(0.61, 120.0)],
            "timestamp": pd.Timestamp("2026-01-23 10:30:00", tz="UTC"),
        }
        intent_data = {"bid_price": 0.595, "bid_size": 30.0, "ask_price": 0.615, "ask_size": 30.0}
        result = build_ladder(book_data, intent_data, pd.DataFrame(), "TEST", 3, 1)
        result_str = str(result)
        assert "0.595" in result_str
        assert "0.615" in result_str

    def test_fill_flash_class_applied(self):
        book_data = {
            "bids": [(0.59, 100.0)],
            "asks": [(0.61, 120.0)],
            "timestamp": pd.Timestamp("2026-01-23 10:30:00", tz="UTC"),
        }
        fills = _fills_df()
        result = build_ladder(book_data, None, fills, "TEST", 2, 1)
        result_str = str(result)
        assert "fill-flash" in result_str
