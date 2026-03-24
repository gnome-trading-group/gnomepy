"""Tests for GenericRecorder, Recorder registry, persistence, and replay utilities."""
import os
import tempfile

import numpy as np
import pytest

from gnomepy.backtest.recorder import (
    GenericRecorder,
    ModelValueRecorder,
    Recorder,
    _BUILTIN_RECORDER_NAMES,
)
from gnomepy.backtest.stats.replay import align_records, compare_runs
from gnomepy.data.types import SchemaType


_DTYPE = np.dtype([
    ('timestamp', 'i8'),
    ('value_a', 'f8'),
    ('value_b', 'f8'),
])

_LISTING_IDS = [1, 2]


# ---------------------------------------------------------------------------
# GenericRecorder
# ---------------------------------------------------------------------------

class TestGenericRecorder:
    def _make(self, size=1000):
        return GenericRecorder(_LISTING_IDS, _DTYPE, size=size)

    def test_log_and_retrieve(self):
        rec = self._make()
        rec.log(1, timestamp=1000, value_a=1.5, value_b=2.5)
        rec.log(1, timestamp=2000, value_a=3.0, value_b=4.0)

        record = rec.get_record(1)
        assert len(record.arr) == 2
        assert record.arr[0]['timestamp'] == 1000
        assert record.arr[0]['value_a'] == pytest.approx(1.5)
        assert record.arr[1]['value_b'] == pytest.approx(4.0)

    def test_dataframe_has_correct_columns(self):
        rec = self._make()
        rec.log(1, timestamp=1000, value_a=1.0, value_b=2.0)
        record = rec.get_record(1)
        df = record.df
        assert 'value_a' in df.columns
        assert 'value_b' in df.columns

    def test_timestamp_becomes_index(self):
        rec = self._make()
        rec.log(1, timestamp=10 ** 9, value_a=0.1, value_b=0.2)
        record = rec.get_record(1)
        assert record.df.index.name == 'timestamp'

    def test_empty_record(self):
        rec = self._make()
        record = rec.get_record(2)
        assert len(record.arr) == 0

    def test_missing_timestamp_raises(self):
        rec = self._make()
        with pytest.raises(ValueError, match="timestamp"):
            rec.log(1, value_a=1.0)

    def test_unknown_listing_raises(self):
        rec = self._make()
        with pytest.raises(KeyError):
            rec.log(999, timestamp=1, value_a=0.0)

    def test_auto_resize(self):
        rec = GenericRecorder(_LISTING_IDS, _DTYPE, size=4, auto_resize=True)
        for i in range(10):
            rec.log(1, timestamp=i, value_a=float(i), value_b=0.0)
        record = rec.get_record(1)
        assert len(record.arr) == 10

    def test_no_resize_raises(self):
        rec = GenericRecorder(_LISTING_IDS, _DTYPE, size=2, auto_resize=False)
        rec.log(1, timestamp=0, value_a=0.0, value_b=0.0)
        rec.log(1, timestamp=1, value_a=0.0, value_b=0.0)
        with pytest.raises(IndexError):
            rec.log(1, timestamp=2, value_a=0.0, value_b=0.0)

    def test_model_value_recorder_alias(self):
        rec = ModelValueRecorder(_LISTING_IDS, _DTYPE)
        assert isinstance(rec, GenericRecorder)

    def test_clear(self):
        rec = self._make()
        rec.log(1, timestamp=1, value_a=1.0, value_b=2.0)
        rec.clear()
        record = rec.get_record(1)
        assert len(record.arr) == 0


# ---------------------------------------------------------------------------
# Recorder registry
# ---------------------------------------------------------------------------

class TestRecorderRegistry:
    def _make_recorder(self):
        return Recorder([1, 2], SchemaType.MBP_10)

    def test_register_and_get(self):
        recorder = self._make_recorder()
        custom = GenericRecorder([1, 2], _DTYPE)
        recorder.register('signals', custom)
        assert recorder.get_custom_recorder('signals') is custom

    def test_get_custom_record(self):
        recorder = self._make_recorder()
        custom = GenericRecorder([1, 2], _DTYPE)
        custom.log(1, timestamp=100, value_a=9.9, value_b=0.0)
        recorder.register('sig', custom)
        record = recorder.get_custom_record('sig', 1)
        assert len(record.arr) == 1

    def test_get_all_custom_recorders(self):
        recorder = self._make_recorder()
        r1 = GenericRecorder([1, 2], _DTYPE)
        r2 = GenericRecorder([1, 2], _DTYPE)
        recorder.register('a', r1)
        recorder.register('b', r2)
        all_recs = recorder.get_all_custom_recorders()
        assert set(all_recs.keys()) == {'a', 'b'}

    def test_register_builtin_name_raises(self):
        recorder = self._make_recorder()
        custom = GenericRecorder([1, 2], _DTYPE)
        for name in _BUILTIN_RECORDER_NAMES:
            with pytest.raises(ValueError, match="built-in"):
                recorder.register(name, custom)

    def test_register_duplicate_name_raises(self):
        recorder = self._make_recorder()
        custom = GenericRecorder([1, 2], _DTYPE)
        recorder.register('x', custom)
        with pytest.raises(ValueError, match="already registered"):
            recorder.register('x', GenericRecorder([1, 2], _DTYPE))

    def test_get_unknown_recorder_raises(self):
        recorder = self._make_recorder()
        with pytest.raises(KeyError):
            recorder.get_custom_recorder('nonexistent')

    def test_clear_includes_custom(self):
        recorder = self._make_recorder()
        custom = GenericRecorder([1, 2], _DTYPE)
        custom.log(1, timestamp=1, value_a=1.0, value_b=2.0)
        recorder.register('c', custom)
        recorder.clear()
        assert len(recorder.get_custom_record('c', 1).arr) == 0

    def test_get_total_record_count_includes_custom(self):
        recorder = self._make_recorder()
        custom = GenericRecorder([1, 2], _DTYPE)
        custom.log(1, timestamp=1, value_a=0.0, value_b=0.0)
        custom.log(2, timestamp=2, value_a=0.0, value_b=0.0)
        recorder.register('c', custom)
        count = recorder.get_total_record_count()
        assert count >= 2

    def test_get_summary_includes_custom(self):
        recorder = self._make_recorder()
        custom = GenericRecorder([1, 2], _DTYPE)
        custom.log(1, timestamp=1, value_a=0.0, value_b=0.0)
        recorder.register('mymod', custom)
        summary = recorder.get_summary()
        assert 'mymod' in summary['custom_recorders']

    def test_repr_shows_custom_count(self):
        recorder = self._make_recorder()
        recorder.register('z', GenericRecorder([1, 2], _DTYPE))
        assert 'custom_recorders=1' in repr(recorder)


# ---------------------------------------------------------------------------
# align_records
# ---------------------------------------------------------------------------

class TestAlignRecords:
    def test_align_two_records(self):
        r1 = GenericRecorder([1], _DTYPE)
        r1.log(1, timestamp=1_000_000_000, value_a=1.0, value_b=0.0)
        r1.log(1, timestamp=3_000_000_000, value_a=3.0, value_b=0.0)

        r2 = GenericRecorder([1], np.dtype([('timestamp', 'i8'), ('score', 'f8')]))
        r2.log(1, timestamp=2_000_000_000, score=99.0)
        r2.log(1, timestamp=3_000_000_000, score=88.0)

        joined = align_records({
            'sig': r1.get_record(1),
            'model': r2.get_record(1),
        })

        assert 'sig.value_a' in joined.columns
        assert 'model.score' in joined.columns
        assert len(joined) == 3  # outer join → 3 distinct timestamps

    def test_column_prefixing(self):
        r1 = GenericRecorder([1], _DTYPE)
        r1.log(1, timestamp=1_000_000_000, value_a=5.0, value_b=0.0)
        joined = align_records({'rec': r1.get_record(1)})
        assert all(c.startswith('rec.') for c in joined.columns)

    def test_empty_records_returns_empty(self):
        result = align_records({})
        assert result.empty


# ---------------------------------------------------------------------------
# compare_runs
# ---------------------------------------------------------------------------

class TestCompareRuns:
    def _make_run_with_market(self, listing_id: int, price: float):
        from gnomepy.backtest.recorder import MarketRecorder, RecordType
        recorder = Recorder([listing_id], SchemaType.MBP_10)
        recorder.market_recorder.log(
            RecordType.MARKET, listing_id, timestamp=1_000_000_000, price=price, quantity=1.0
        )
        return recorder

    def test_compare_two_runs(self):
        run_a = self._make_run_with_market(1, 100.0)
        run_b = self._make_run_with_market(1, 200.0)

        result = compare_runs({'a': run_a, 'b': run_b}, listing_id=1, recorder_names=['market'])
        assert any('a.market.' in c for c in result.columns)
        assert any('b.market.' in c for c in result.columns)

    def test_defaults_to_all_recorders(self):
        run_a = self._make_run_with_market(1, 100.0)
        custom = GenericRecorder([1], _DTYPE)
        custom.log(1, timestamp=1_000_000_000, value_a=7.0, value_b=0.0)
        run_a.register('sig', custom)

        result = compare_runs({'run': run_a}, listing_id=1)
        assert any('run.market.' in c for c in result.columns)
        assert any('run.sig.' in c for c in result.columns)

    def test_empty_runs_returns_empty(self):
        result = compare_runs({}, listing_id=1)
        assert result.empty


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

class TestPersistenceRoundTrip:
    def test_custom_recorder_survives_npz(self):
        recorder = Recorder([1, 2], SchemaType.MBP_10)
        custom = GenericRecorder([1, 2], _DTYPE)
        custom.log(1, timestamp=500, value_a=3.14, value_b=2.71)
        custom.log(1, timestamp=600, value_a=1.41, value_b=1.73)
        custom.log(2, timestamp=700, value_a=0.0, value_b=9.9)
        recorder.register('model', custom)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name

        try:
            recorder.to_npz(path)
            loaded = Recorder.from_npz(path, SchemaType.MBP_10)

            assert 'model' in loaded.custom_recorders
            r1 = loaded.get_custom_record('model', 1)
            assert len(r1.arr) == 2
            assert r1.arr[0]['value_a'] == pytest.approx(3.14)
            assert r1.arr[1]['value_a'] == pytest.approx(1.41)

            r2 = loaded.get_custom_record('model', 2)
            assert len(r2.arr) == 1
            assert r2.arr[0]['value_b'] == pytest.approx(9.9)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Stats strict param (regression)
# ---------------------------------------------------------------------------

class TestStatsStrict:
    def test_market_record_stats_strict(self):
        from gnomepy.backtest.recorder import MarketRecorder, RecordType
        mr = MarketRecorder([1], SchemaType.MBP_10)
        mr.log(RecordType.MARKET, 1, timestamp=1_000_000_000, price=100.0, quantity=1.0)
        mr.log(RecordType.MARKET, 1, timestamp=2_000_000_000, price=101.0, quantity=1.0)
        record = mr.get_record(1)
        stats = record.stats()
        assert stats is not None

    def test_generic_record_stats_non_strict(self):
        rec = GenericRecorder([1], _DTYPE)
        rec.log(1, timestamp=1_000_000_000, value_a=1.0, value_b=2.0)
        rec.log(1, timestamp=2_000_000_000, value_a=3.0, value_b=4.0)
        record = rec.get_record(1)
        stats = record.stats()
        assert stats is not None
