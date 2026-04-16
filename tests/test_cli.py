"""Unit tests for CLI helper functions — no JVM needed."""
from __future__ import annotations

import re

from gnomepy.utils import uuid7 as _uuid7


_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


class TestUuid7:
    def test_format(self):
        uid = _uuid7()
        assert _UUID_RE.match(uid), f"UUIDv7 format mismatch: {uid!r}"

    def test_version_nibble(self):
        uid = _uuid7()
        assert uid[14] == "7"

    def test_variant_bits(self):
        uid = _uuid7()
        variant_char = uid[19]
        assert variant_char in "89ab"

    def test_uniqueness(self):
        ids = {_uuid7() for _ in range(50)}
        assert len(ids) == 50

    def test_timestamp_non_decreasing(self):
        # The first 12 hex chars encode the 48-bit ms timestamp. Calls separated
        # by a sleep should always produce a higher or equal timestamp prefix.
        import time
        uid_a = _uuid7()
        time.sleep(0.002)  # 2 ms — crosses a timestamp boundary
        uid_b = _uuid7()
        ts_a = uid_a.replace("-", "")[:12]
        ts_b = uid_b.replace("-", "")[:12]
        assert ts_a <= ts_b
