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
from gnomepy.java.backtest.config import (
    BacktestConfig,
    ExchangeProfileConfig,
    GaussianLatencyConfig,
    ListingSimConfig,
    OptimisticQueueConfig,
    ProbabilisticQueueConfig,
    RiskAverseQueueConfig,
    RiskConfig,
    StaticFeeConfig,
    StaticLatencyConfig,
    StrategyConfig,
)
from gnomepy.java.backtest.orders import ExecutionReport
from gnomepy.java.backtest.runner import Backtest, run_backtest
from gnomepy.java.cache import MarketDataCache
from gnomepy.java.backtest.strategy import Strategy
from gnomepy.java.datastore import DataStore
from gnomepy.java.enums import (
    Action,
    ExecType,
    OrderStatus,
    OrderType,
    SchemaType,
    Side,
    TimeInForce,
)
from gnomepy.java.oms import (
    Intent,
    OmsView,
    PositionInfo,
    TrackedOrderInfo,
)
from gnomepy.java.recorder import BacktestResults
from gnomepy.java.schemas import (
    Bbo1mSchema,
    Bbo1sSchema,
    BboSchema,
    MboSchema,
    Mbp1Schema,
    Mbp10Schema,
    Ohlcv1hSchema,
    Ohlcv1mSchema,
    Ohlcv1sSchema,
    OhlcvSchema,
    Schema,
    TradesSchema,
    wrap_schema,
)
from gnomepy.java.statics import Scales
from gnomepy.metadata import BacktestMetadata
from gnomepy.reporting import BacktestReport
from gnomepy.reporting.metrics import Curves, build_curves, compute_sharpe
from gnomepy.reporting.plots import ReportSection
from gnomepy.utils import generate_backtest_id, uuid7

__all__ = [
    # Top-level API
    "Strategy",
    "Backtest",
    "run_backtest",
    "MarketDataCache",
    # Backtest config
    "BacktestConfig",
    "ListingSimConfig",
    "ExchangeProfileConfig",
    "StrategyConfig",
    "RiskConfig",
    "StaticFeeConfig",
    "StaticLatencyConfig",
    "GaussianLatencyConfig",
    "OptimisticQueueConfig",
    "RiskAverseQueueConfig",
    "ProbabilisticQueueConfig",
    # Orders / OMS
    "Intent",
    "ExecutionReport",
    "OmsView",
    "PositionInfo",
    "TrackedOrderInfo",
    "BacktestResults",
    # Enums
    "SchemaType",
    "Side",
    "Action",
    "OrderType",
    "TimeInForce",
    "ExecType",
    "OrderStatus",
    # Schemas
    "Schema",
    "MboSchema",
    "Mbp10Schema",
    "Mbp1Schema",
    "BboSchema",
    "Bbo1sSchema",
    "Bbo1mSchema",
    "TradesSchema",
    "OhlcvSchema",
    "Ohlcv1sSchema",
    "Ohlcv1mSchema",
    "Ohlcv1hSchema",
    "wrap_schema",
    "DataStore",
    "Scales",
    # Metadata
    "BacktestMetadata",
    "generate_backtest_id",
    "uuid7",
    # Reporting
    "BacktestReport",
    "Curves",
    "build_curves",
    "compute_sharpe",
    "ReportSection",
]
