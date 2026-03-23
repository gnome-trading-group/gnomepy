from __future__ import annotations


def _get_static(field_name: str, static_type: type) -> int:
    """Lazily read a constant from group.gnometrading.schemas.Statics."""
    import jpype
    from gnomepy.java._jvm import ensure_jvm_started

    ensure_jvm_started()
    Statics = jpype.JClass("group.gnometrading.schemas.Statics")

    value = getattr(Statics, field_name)
    return static_type(value)


class _LazyStatic:
    """Descriptor that resolves a Java static constant on first access."""

    def __init__(self, field_name: str, static_type: type):
        self._field = field_name
        self._type = static_type
        self._value: int | None = None

    def __get__(self, obj, objtype=None) -> int:
        if self._value is None:
            self._value = _get_static(self._field, self._type)
        return self._value


class Scales:
    """Scaling factors from group.gnometrading.schemas.Statics.

    Usage:
        from gnomepy.java.statics import Scales
        raw_price / Scales.PRICE  # human-readable price
        raw_size / Scales.SIZE    # human-readable size
    """

    PRICE = _LazyStatic("PRICE_SCALING_FACTOR", int)
    SIZE = _LazyStatic("SIZE_SCALING_FACTOR", int)
