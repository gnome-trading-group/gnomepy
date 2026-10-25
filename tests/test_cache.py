"""Unit tests for MarketDataCache — no JVM required."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from gnomepy.java.cache import MarketDataCache


class TestMarketDataCache:
    def test_put_and_get(self, tmp_path):
        cache = MarketDataCache(tmp_path)
        cache.put("bucket", "foo/bar/baz.sbe", b"hello world")
        assert cache.get("bucket", "foo/bar/baz.sbe") == b"hello world"

    def test_get_missing_returns_none(self, tmp_path):
        cache = MarketDataCache(tmp_path)
        assert cache.get("bucket", "nonexistent/key") is None

    def test_has_false_before_put(self, tmp_path):
        cache = MarketDataCache(tmp_path)
        assert not cache.has("bucket", "foo")

    def test_has_true_after_put(self, tmp_path):
        cache = MarketDataCache(tmp_path)
        cache.put("bucket", "foo", b"data")
        assert cache.has("bucket", "foo")

    def test_empty_file_treated_as_corrupt(self, tmp_path):
        cache = MarketDataCache(tmp_path)
        path = cache._key_to_path("bucket", "corrupt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        assert cache.get("bucket", "corrupt") is None
        assert not path.exists()

    def test_clear_all(self, tmp_path):
        cache = MarketDataCache(tmp_path)
        cache.put("bucket", "a/b", b"1")
        cache.put("bucket", "c/d", b"2")
        count = cache.clear()
        assert count == 2
        assert cache.get("bucket", "a/b") is None
        assert cache.get("bucket", "c/d") is None

    def test_clear_with_prefix(self, tmp_path):
        cache = MarketDataCache(tmp_path)
        cache.put("bucket", "mbo/1/a", b"1")
        cache.put("bucket", "mbo/1/b", b"2")
        cache.put("bucket", "mbo/2/c", b"3")
        count = cache.clear(prefix="bucket/mbo/1")
        assert count == 2
        assert cache.get("bucket", "mbo/1/a") is None
        assert cache.get("bucket", "mbo/1/b") is None
        assert cache.has("bucket", "mbo/2/c")

    def test_clear_nonexistent_prefix(self, tmp_path):
        cache = MarketDataCache(tmp_path)
        assert cache.clear(prefix="nope") == 0

    def test_size_empty(self, tmp_path):
        cache = MarketDataCache(tmp_path)
        total, count = cache.size()
        assert total == 0
        assert count == 0

    def test_size_with_files(self, tmp_path):
        cache = MarketDataCache(tmp_path)
        cache.put("bucket", "a", b"12345")
        cache.put("bucket", "b", b"678")
        total, count = cache.size()
        assert count == 2
        assert total == 8

    def test_key_to_path_preserves_structure(self, tmp_path):
        cache = MarketDataCache(tmp_path)
        path = cache._key_to_path("bucket", "mbo/1/2/2024/01/15/09/30.sbe.zst")
        assert path == tmp_path / "bucket" / "mbo" / "1" / "2" / "2024" / "01" / "15" / "09" / "30.sbe.zst"

    def test_key_to_path_strips_leading_slash(self, tmp_path):
        cache = MarketDataCache(tmp_path)
        path = cache._key_to_path("bucket", "/foo/bar")
        assert path == tmp_path / "bucket" / "foo" / "bar"

    def test_atomic_write_no_tmp_remnants(self, tmp_path):
        cache = MarketDataCache(tmp_path)
        cache.put("bucket", "key", b"data")
        tmp_files = list(tmp_path.rglob("*.tmp"))
        assert len(tmp_files) == 0

    def test_path_traversal_rejected(self, tmp_path):
        cache = MarketDataCache(tmp_path)
        with pytest.raises(ValueError, match="escapes cache root"):
            cache._key_to_path("bucket", "../../etc/passwd")

    def test_default_cache_dir(self):
        cache = MarketDataCache()
        assert cache._root == Path.home() / ".gnomepy" / "cache" / "market_data"

    def test_env_override(self, tmp_path, monkeypatch):
        custom = tmp_path / "custom"
        monkeypatch.setenv("GNOMEPY_CACHE_DIR", str(custom))
        cache = MarketDataCache()
        assert cache._root == custom

    def test_explicit_path_overrides_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GNOMEPY_CACHE_DIR", str(tmp_path / "env_dir"))
        explicit = tmp_path / "explicit"
        cache = MarketDataCache(explicit)
        assert cache._root == explicit

    def test_same_key_in_two_buckets_is_two_entries(self, tmp_path):
        cache = MarketDataCache(tmp_path)
        cache.put("gnome-market-data-dev", "7/book/mbp-10/2026/01/01/00/00.zst", b"dev")
        cache.put("gnome-market-data-prod", "7/book/mbp-10/2026/01/01/00/00.zst", b"prod")
        assert cache.get("gnome-market-data-dev", "7/book/mbp-10/2026/01/01/00/00.zst") == b"dev"
        assert cache.get("gnome-market-data-prod", "7/book/mbp-10/2026/01/01/00/00.zst") == b"prod"

    def test_overwrite_existing_key(self, tmp_path):
        cache = MarketDataCache(tmp_path)
        cache.put("bucket", "key", b"first")
        cache.put("bucket", "key", b"second")
        assert cache.get("bucket", "key") == b"second"
