"""gnomepy — backtesting infrastructure for the gnome trading system.

Public API:

    from gnomepy import (
        Strategy, Backtest, run_backtest,
        BacktestConfig, ListingSimConfig, ExchangeProfileConfig,
        StrategyConfig, RiskConfig,
        StaticFeeConfig, StaticLatencyConfig, GaussianLatencyConfig,
        OptimisticQueueConfig, RiskAverseQueueConfig, ProbabilisticQueueConfig,
        Intent, ExecutionReport, OmsView,
        SchemaType, Side, Action,
        Mbp10Schema, ...
    )
"""
import importlib

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BacktestConfig": ("gnomepy.java.backtest.config", "BacktestConfig"),
    "ExchangeProfileConfig": ("gnomepy.java.backtest.config", "ExchangeProfileConfig"),
    "GaussianLatencyConfig": ("gnomepy.java.backtest.config", "GaussianLatencyConfig"),
    "ListingSimConfig": ("gnomepy.java.backtest.config", "ListingSimConfig"),
    "OptimisticQueueConfig": ("gnomepy.java.backtest.config", "OptimisticQueueConfig"),
    "ProbabilisticQueueConfig": ("gnomepy.java.backtest.config", "ProbabilisticQueueConfig"),
    "RiskAverseQueueConfig": ("gnomepy.java.backtest.config", "RiskAverseQueueConfig"),
    "RiskConfig": ("gnomepy.java.backtest.config", "RiskConfig"),
    "StaticFeeConfig": ("gnomepy.java.backtest.config", "StaticFeeConfig"),
    "StaticLatencyConfig": ("gnomepy.java.backtest.config", "StaticLatencyConfig"),
    "StrategyConfig": ("gnomepy.java.backtest.config", "StrategyConfig"),
    "ExecutionReport": ("gnomepy.java.backtest.orders", "ExecutionReport"),
    "Backtest": ("gnomepy.java.backtest.runner", "Backtest"),
    "run_backtest": ("gnomepy.java.backtest.runner", "run_backtest"),
    "MarketDataCache": ("gnomepy.java.cache", "MarketDataCache"),
    "Strategy": ("gnomepy.java.backtest.strategy", "Strategy"),
    "DataStore": ("gnomepy.java.datastore", "DataStore"),
    "Action": ("gnomepy.java.enums", "Action"),
    "ExecType": ("gnomepy.java.enums", "ExecType"),
    "OrderStatus": ("gnomepy.java.enums", "OrderStatus"),
    "OrderType": ("gnomepy.java.enums", "OrderType"),
    "SchemaType": ("gnomepy.java.enums", "SchemaType"),
    "Side": ("gnomepy.java.enums", "Side"),
    "TimeInForce": ("gnomepy.java.enums", "TimeInForce"),
    "Intent": ("gnomepy.java.oms", "Intent"),
    "OmsView": ("gnomepy.java.oms", "OmsView"),
    "PositionInfo": ("gnomepy.java.oms", "PositionInfo"),
    "TrackedOrderInfo": ("gnomepy.java.oms", "TrackedOrderInfo"),
    "BacktestResults": ("gnomepy.java.recorder", "BacktestResults"),
    "Bbo1mSchema": ("gnomepy.java.schemas", "Bbo1mSchema"),
    "Bbo1sSchema": ("gnomepy.java.schemas", "Bbo1sSchema"),
    "BboSchema": ("gnomepy.java.schemas", "BboSchema"),
    "MboSchema": ("gnomepy.java.schemas", "MboSchema"),
    "Mbp1Schema": ("gnomepy.java.schemas", "Mbp1Schema"),
    "Mbp10Schema": ("gnomepy.java.schemas", "Mbp10Schema"),
    "Ohlcv1hSchema": ("gnomepy.java.schemas", "Ohlcv1hSchema"),
    "Ohlcv1mSchema": ("gnomepy.java.schemas", "Ohlcv1mSchema"),
    "Ohlcv1sSchema": ("gnomepy.java.schemas", "Ohlcv1sSchema"),
    "OhlcvSchema": ("gnomepy.java.schemas", "OhlcvSchema"),
    "Schema": ("gnomepy.java.schemas", "Schema"),
    "TradesSchema": ("gnomepy.java.schemas", "TradesSchema"),
    "wrap_schema": ("gnomepy.java.schemas", "wrap_schema"),
    "Scales": ("gnomepy.java.statics", "Scales"),
    "BacktestMetadata": ("gnomepy.metadata", "BacktestMetadata"),
    "BacktestReport": ("gnomepy.reporting", "BacktestReport"),
    "Curves": ("gnomepy.reporting.metrics", "Curves"),
    "build_curves": ("gnomepy.reporting.metrics", "build_curves"),
    "compute_sharpe": ("gnomepy.reporting.metrics", "compute_sharpe"),
    "ReportSection": ("gnomepy.reporting.plots", "ReportSection"),
    "generate_backtest_id": ("gnomepy.utils", "generate_backtest_id"),
    "uuid7": ("gnomepy.utils", "uuid7"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        mod_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(mod_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_IMPORTS.keys())
