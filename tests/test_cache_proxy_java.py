"""The caching S3 proxy around the Java engine's reads — requires the gnome-backtest uber JAR; skipped without it."""
from __future__ import annotations

import jpype
import pytest
import zstandard
from jpype import JImplements, JOverride

from gnomepy.java._classpath import discover_classpath
from gnomepy.java._jvm import ensure_jvm_started
from gnomepy.java.cache import MarketDataCache, create_caching_s3_proxy
from gnomepy.java.schemas import Mbp10Schema

BUCKET = "gnome-market-data-test"


def _jar_available() -> bool:
    try:
        return len(discover_classpath("gnome-backtest")) > 0
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not _jar_available(), reason="gnome-backtest JAR not available")


@pytest.fixture(scope="module", autouse=True)
def _jvm():
    ensure_jvm_started()


class _CountingS3:
    """A Java S3Client serving getObject from a dict, counting the reads that reach it."""

    def __init__(self, objects: dict[str, bytes]):
        self.reads = 0
        GetObjectRequest = jpype.JClass("software.amazon.awssdk.services.s3.model.GetObjectRequest")
        GetObjectResponse = jpype.JClass("software.amazon.awssdk.services.s3.model.GetObjectResponse")
        ResponseInputStream = jpype.JClass("software.amazon.awssdk.core.ResponseInputStream")
        AbortableInputStream = jpype.JClass("software.amazon.awssdk.http.AbortableInputStream")
        ByteArrayInputStream = jpype.JClass("java.io.ByteArrayInputStream")
        outer = self

        @JImplements("java.lang.reflect.InvocationHandler")
        class _Handler:
            @JOverride
            def invoke(self, proxy, method, args):
                request = args[0]
                if not isinstance(request, GetObjectRequest):
                    builder = GetObjectRequest.builder()
                    request.accept(builder)
                    request = builder.build()
                outer.reads += 1
                data = jpype.JArray(jpype.JByte)(objects[str(request.key())])
                return ResponseInputStream(
                    GetObjectResponse.builder().build(), AbortableInputStream.create(ByteArrayInputStream(data)))

        S3Client = jpype.JClass("software.amazon.awssdk.services.s3.S3Client")
        self.client = jpype.JClass("java.lang.reflect.Proxy").newProxyInstance(
            S3Client.class_.getClassLoader(), [S3Client.class_], _Handler())


def test_the_engines_reads_are_served_from_the_cache_after_the_first(tmp_path):
    StreamEntry = jpype.JClass("group.gnometrading.data.StreamEntry")
    RecordType = jpype.JClass("group.gnometrading.schemas.RecordType")
    minute = jpype.JClass("java.time.LocalDateTime").of(2026, 1, 1, 0, 0)
    entry = StreamEntry(7, RecordType.MBP_10, minute)
    record = Mbp10Schema(stream_id=7, timestamp_event=1, timestamp_recv=2, action="Add", side="Bid")
    real = _CountingS3({str(entry.getKey()): zstandard.ZstdCompressor().compress(record.encode())})
    caching = create_caching_s3_proxy(real.client, MarketDataCache(tmp_path), BUCKET)

    first = entry.loadFromS3(caching, BUCKET)
    second = entry.loadFromS3(caching, BUCKET)

    assert real.reads == 1
    assert [int(r.streamId()) for r in first] == [int(r.streamId()) for r in second] == [7]
