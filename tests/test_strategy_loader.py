"""Unit tests for _load_python_strategy — no JVM needed."""
from __future__ import annotations

import pytest

from gnomepy.java.backtest.runner import _load_python_strategy
from gnomepy.java.backtest.strategy import Strategy


class _FakeStrategy(Strategy):
    def __init__(self, x: int = 0, label: str = ""):
        self.x = x
        self.label = label

    def on_market_data(self, data):
        return []

    def on_execution_report(self, report):
        return []


class _NotAStrategy:
    pass


class TestLoadPythonStrategy:
    def test_valid_import(self, monkeypatch):
        import sys
        import types

        mod = types.ModuleType("fake_strategies")
        mod.MyStrat = _FakeStrategy
        monkeypatch.setitem(sys.modules, "fake_strategies", mod)

        instance = _load_python_strategy("fake_strategies:MyStrat")
        assert isinstance(instance, _FakeStrategy)

    def test_kwargs_passed(self, monkeypatch):
        import sys
        import types

        mod = types.ModuleType("fake_strategies2")
        mod.MyStrat = _FakeStrategy
        monkeypatch.setitem(sys.modules, "fake_strategies2", mod)

        instance = _load_python_strategy("fake_strategies2:MyStrat", {"x": 7, "label": "hi"})
        assert instance.x == 7
        assert instance.label == "hi"

    def test_missing_colon_raises(self):
        with pytest.raises(ValueError, match="module.path:ClassName"):
            _load_python_strategy("no_colon_here")

    def test_missing_module_raises(self):
        with pytest.raises((ModuleNotFoundError, ImportError)):
            _load_python_strategy("nonexistent.module.xyz:SomeClass")

    def test_non_strategy_class_raises(self, monkeypatch):
        import sys
        import types

        mod = types.ModuleType("fake_strategies3")
        mod.NotAStrat = _NotAStrategy
        monkeypatch.setitem(sys.modules, "fake_strategies3", mod)

        with pytest.raises(ValueError, match="gnomepy.Strategy"):
            _load_python_strategy("fake_strategies3:NotAStrat")
