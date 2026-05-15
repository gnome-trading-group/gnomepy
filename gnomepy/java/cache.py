from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".gnomepy" / "cache" / "market_data"


class MarketDataCache:
    """Local disk cache for raw S3 market data objects.

    Keys are S3 object keys (e.g. ``mbo/1/2/2024/01/15/09/30.sbe.zst``),
    stored as-is under the cache root directory. Market data is immutable
    historical data so no TTL is needed; use :meth:`clear` to evict manually.
    """

    def __init__(self, cache_dir: Path | str | None = None):
        if cache_dir is None:
            env_dir = os.environ.get("GNOMEPY_CACHE_DIR")
            cache_dir = env_dir if env_dir else _DEFAULT_CACHE_DIR
        self._root = Path(cache_dir)

    def _key_to_path(self, s3_key: str) -> Path:
        clean = s3_key.lstrip("/")
        path = (self._root / clean).resolve()
        if not str(path).startswith(str(self._root.resolve())):
            raise ValueError(f"S3 key escapes cache root: {s3_key!r}")
        return self._root / clean

    def get(self, s3_key: str) -> bytes | None:
        path = self._key_to_path(s3_key)
        if not path.is_file():
            return None
        data = path.read_bytes()
        if not data:
            path.unlink(missing_ok=True)
            return None
        return data

    def put(self, s3_key: str, data: bytes) -> Path:
        path = self._key_to_path(s3_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return path

    def has(self, s3_key: str) -> bool:
        path = self._key_to_path(s3_key)
        return path.is_file() and path.stat().st_size > 0

    def clear(self, prefix: str | None = None) -> int:
        target = (self._root / prefix) if prefix else self._root
        if not target.exists():
            return 0
        count = sum(1 for p in target.rglob("*") if p.is_file())
        shutil.rmtree(target)
        return count

    def size(self) -> tuple[int, int]:
        if not self._root.exists():
            return (0, 0)
        total = 0
        count = 0
        for p in self._root.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
                count += 1
        return (total, count)


def create_caching_s3_proxy(real_s3_client, cache: MarketDataCache, bucket: str):
    """Wrap a Java S3Client with a local disk caching layer.

    Uses ``java.lang.reflect.Proxy`` to intercept ``getObject`` and
    ``getObjectAsBytes`` calls, serving from ``cache`` on hits and
    persisting downloaded bytes on misses. All other S3 methods are
    delegated transparently to ``real_s3_client``.
    """
    import jpype
    from jpype import JImplements, JOverride

    Proxy = jpype.JClass("java.lang.reflect.Proxy")
    S3Client = jpype.JClass("software.amazon.awssdk.services.s3.S3Client")

    @JImplements("java.lang.reflect.InvocationHandler")
    class _CachingHandler:
        @JOverride
        def invoke(self, proxy, method, args):
            name = str(method.getName())

            if name in ("getObjectAsBytes", "getObject") and args is not None and len(args) >= 1:
                request = args[0]
                try:
                    key = str(request.key())
                    req_bucket = str(request.bucket())
                except Exception:
                    return method.invoke(real_s3_client, args)

                if req_bucket == bucket:
                    cached = cache.get(key)
                    if cached is not None:
                        logger.debug("market data cache hit: %s", key)
                        if name == "getObjectAsBytes":
                            return _response_bytes_from_cache(cached)
                        else:
                            return _response_stream_from_cache(cached)

                    # Cache miss — call real client, then persist
                    result = method.invoke(real_s3_client, args)
                    if name == "getObjectAsBytes":
                        raw = bytes(result.asByteArray())
                        cache.put(key, raw)
                        return result
                    else:
                        # ResponseInputStream can only be read once
                        raw = bytes(result.readAllBytes())
                        result.close()
                        cache.put(key, raw)
                        return _response_stream_from_cache(raw)

            if args is None:
                return method.invoke(real_s3_client, jpype.JArray(jpype.JObject)(0))
            return method.invoke(real_s3_client, args)

    handler = _CachingHandler()
    interfaces = jpype.JArray(jpype.JClass("java.lang.Class"))([S3Client.class_])
    return Proxy.newProxyInstance(S3Client.class_.getClassLoader(), interfaces, handler)


def _response_bytes_from_cache(data: bytes):
    import jpype

    GetObjectResponse = jpype.JClass("software.amazon.awssdk.services.s3.model.GetObjectResponse")
    ResponseBytes = jpype.JClass("software.amazon.awssdk.core.ResponseBytes")

    response = GetObjectResponse.builder().contentLength(jpype.JLong(len(data))).build()
    java_bytes = jpype.JArray(jpype.JByte)(data)
    return ResponseBytes.fromByteArrayUnsafe(response, java_bytes)


def _response_stream_from_cache(data: bytes):
    import jpype

    GetObjectResponse = jpype.JClass("software.amazon.awssdk.services.s3.model.GetObjectResponse")
    ResponseInputStream = jpype.JClass("software.amazon.awssdk.core.ResponseInputStream")
    ByteArrayInputStream = jpype.JClass("java.io.ByteArrayInputStream")
    AbortableInputStream = jpype.JClass("software.amazon.awssdk.http.AbortableInputStream")

    response = GetObjectResponse.builder().contentLength(jpype.JLong(len(data))).build()
    java_bytes = jpype.JArray(jpype.JByte)(data)
    bais = ByteArrayInputStream(java_bytes)
    abortable = AbortableInputStream.create(bais)
    return ResponseInputStream(response, abortable)
