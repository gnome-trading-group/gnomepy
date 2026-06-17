"""Unit tests for the importer module — no JVM required."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from gnomepy.importer.chunker import chunk_by_minute
from gnomepy.importer.mapping import FieldMapping, ImportConfig
from gnomepy.importer.scaling import (
    apply_size_transform,
    apply_timestamp_transform,
    parse_timestamp_ns,
    scale_price,
    scale_size,
)
from gnomepy.importer.uploader import build_s3_key, default_merged_bucket
from gnomepy.importer.validators import validate
from gnomepy.java.enums import SchemaType


# ---------------------------------------------------------------------------
# scaling.py
# ---------------------------------------------------------------------------

class TestScalePrice:
    def test_round_trip(self):
        assert scale_price(100.0) == 100_000_000_000

    def test_fractional(self):
        assert scale_price(1.23456789) == 1_234_567_890

    def test_zero(self):
        assert scale_price(0) == 0

    def test_integer_input(self):
        assert scale_price(50) == 50_000_000_000


class TestScaleSize:
    def test_whole_number(self):
        assert scale_size(10.0) == 10_000_000

    def test_fractional(self):
        assert scale_size(0.5) == 500_000

    def test_zero(self):
        assert scale_size(0) == 0


class TestParseTimestampNs:
    def test_epoch_ns(self):
        assert parse_timestamp_ns(1_700_000_000_000_000_000, "epoch_ns") == 1_700_000_000_000_000_000

    def test_epoch_us(self):
        assert parse_timestamp_ns(1_700_000_000_000_000, "epoch_us") == 1_700_000_000_000_000_000

    def test_epoch_ms(self):
        assert parse_timestamp_ns(1_700_000_000_000, "epoch_ms") == 1_700_000_000_000_000_000

    def test_epoch_s(self):
        assert parse_timestamp_ns(1_700_000_000, "epoch_s") == 1_700_000_000_000_000_000

    def test_iso8601_utc(self):
        ns = parse_timestamp_ns("2024-01-15T09:30:00Z", "iso8601")
        # 2024-01-15 09:30 UTC
        expected = int(datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc).timestamp() * 1e9)
        assert ns == expected

    def test_iso8601_with_tz(self):
        ns_utc = parse_timestamp_ns("2024-01-15T09:30:00Z", "iso8601")
        ns_tz = parse_timestamp_ns("2024-01-15T04:30:00", "iso8601", tz="US/Eastern")
        assert ns_utc == ns_tz


class TestApplyTimestampTransform:
    def test_epoch_ms_series(self):
        s = pd.Series([1_700_000_000_000, 1_700_000_060_000])
        result = apply_timestamp_transform(s, "epoch_ms", None)
        assert result.iloc[0] == 1_700_000_000_000_000_000
        assert result.iloc[1] == 1_700_000_060_000_000_000

    def test_epoch_ns_series(self):
        values = [1_700_000_000_000_000_000, 1_700_000_001_000_000_000]
        s = pd.Series(values)
        result = apply_timestamp_transform(s, "epoch_ns", None)
        assert list(result) == values


class TestApplySizeTransform:
    def test_basic(self):
        s = pd.Series([1.0, 2.5, 0.0])
        result = apply_size_transform(s)
        assert result.iloc[0] == 1_000_000
        assert result.iloc[1] == 2_500_000
        assert result.iloc[2] == 0


# ---------------------------------------------------------------------------
# chunker.py
# ---------------------------------------------------------------------------

class TestChunkByMinute:
    def _make_df_and_ts(self, timestamps_ns: list[int]) -> tuple[pd.DataFrame, pd.Series]:
        df = pd.DataFrame({"value": range(len(timestamps_ns))})
        ts = pd.Series(timestamps_ns, dtype="int64")
        return df, ts

    def test_single_minute(self):
        base = 1_700_000_000_000_000_000  # some ns epoch
        ts = [base, base + 1_000_000_000, base + 30_000_000_000]
        df, ts_ns = self._make_df_and_ts(ts)
        chunks = chunk_by_minute(df, ts_ns)
        assert len(chunks) == 1

    def test_two_minutes(self):
        # Two records in minute 0, one in minute 1
        minute_ns = 60_000_000_000
        base = 1_700_000_000_000_000_000
        ts = [base, base + 10_000_000_000, base + minute_ns]
        df, ts_ns = self._make_df_and_ts(ts)
        chunks = chunk_by_minute(df, ts_ns)
        assert len(chunks) == 2

    def test_chunk_sizes(self):
        minute_ns = 60_000_000_000
        base = 1_700_000_000_000_000_000
        ts = [base] * 3 + [base + minute_ns] * 2
        df, ts_ns = self._make_df_and_ts(ts)
        chunks = chunk_by_minute(df, ts_ns)
        sizes = sorted(len(v) for v in chunks.values())
        assert sizes == [2, 3]

    def test_keys_are_naive_utc_datetimes(self):
        base = 1_700_000_000_000_000_000
        df, ts_ns = self._make_df_and_ts([base])
        chunks = chunk_by_minute(df, ts_ns)
        key = list(chunks.keys())[0]
        assert isinstance(key, datetime)
        assert key.tzinfo is None

    def test_records_sorted_within_chunk(self):
        base = 1_700_000_000_000_000_000
        ts = [base + 30_000_000_000, base + 10_000_000_000, base]
        df = pd.DataFrame({"value": [30, 10, 0]})
        ts_ns = pd.Series(ts, dtype="int64")
        chunks = chunk_by_minute(df, ts_ns)
        chunk = list(chunks.values())[0]
        assert list(chunk["value"]) == [0, 10, 30]


# ---------------------------------------------------------------------------
# validators.py
# ---------------------------------------------------------------------------

class TestValidate:
    def _base_config(self) -> ImportConfig:
        return ImportConfig(
            schema_type=SchemaType.TRADES,
            security_id=1,
            exchange_id=5,
            timestamp_field="timestamp_event",
            field_mappings=[
                FieldMapping("ts", "timestamp_event", transform="timestamp", timestamp_format="epoch_ms"),
                FieldMapping("px", "price", transform="price"),
                FieldMapping("qty", "size", transform="size"),
            ],
        )

    def _base_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "ts": [1_700_000_000_000, 1_700_000_001_000],
            "px": [50000.0, 50001.0],
            "qty": [0.1, 0.2],
        })

    def test_valid_config(self):
        assert validate(self._base_config(), self._base_df()) == []

    def test_missing_source_column(self):
        config = self._base_config()
        df = self._base_df().drop(columns=["px"])
        errors = validate(config, df)
        assert any("px" in e for e in errors)

    def test_missing_timestamp_mapping(self):
        config = self._base_config()
        config.field_mappings = [m for m in config.field_mappings if m.target_field != "timestamp_event"]
        errors = validate(config, self._base_df())
        assert any("timestamp_field" in e for e in errors)

    def test_timestamp_missing_format(self):
        config = self._base_config()
        config.field_mappings[0].timestamp_format = None
        errors = validate(config, self._base_df())
        assert any("timestamp_format" in e for e in errors)

    def test_enum_missing_map(self):
        config = self._base_config()
        config.field_mappings.append(FieldMapping("side", "side", transform="enum"))
        df = self._base_df()
        df["side"] = ["buy", "sell"]
        errors = validate(config, df)
        assert any("enum_map" in e for e in errors)

    def test_null_timestamps_flagged(self):
        config = self._base_config()
        df = self._base_df()
        df.loc[0, "ts"] = None
        errors = validate(config, df)
        assert any("null" in e for e in errors)


# ---------------------------------------------------------------------------
# uploader.py
# ---------------------------------------------------------------------------

class TestBuildS3Key:
    def test_format_matches_market_data_entry(self):
        dt = datetime(2024, 4, 15, 14, 30)
        key = build_s3_key(532, 151, SchemaType.MBP_10, dt)
        assert key == "532/151/2024/4/15/14/30/mbp-10.zst"

    def test_no_zero_padding(self):
        dt = datetime(2025, 1, 5, 9, 3)
        key = build_s3_key(1, 2, SchemaType.TRADES, dt)
        assert key == "1/2/2025/1/5/9/3/trades.zst"


class TestDefaultMergedBucket:
    def test_uses_stage_env(self, monkeypatch):
        monkeypatch.setenv("STAGE", "dev")
        assert default_merged_bucket() == "gnome-market-data-merged-dev"

    def test_defaults_to_prod(self, monkeypatch):
        monkeypatch.delenv("STAGE", raising=False)
        assert default_merged_bucket() == "gnome-market-data-merged-prod"

    def test_lowercases_stage(self, monkeypatch):
        monkeypatch.setenv("STAGE", "PROD")
        assert default_merged_bucket() == "gnome-market-data-merged-prod"
