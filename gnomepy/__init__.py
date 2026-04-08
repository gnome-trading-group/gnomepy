"""gnomepy — backtesting infrastructure for the gnome trading system.

Public API:

    from gnomepy import (
        Strategy, Backtest, run_backtest,
        Intent, ExecutionReport, OmsView, RiskConfig,
        ExchangeConfig, StaticFeeConfig, StaticLatencyConfig,
        SchemaType, Side, Action,
        Mbp10Schema, ...
    )
"""
from gnomepy.java.backtest.config import (
    ExchangeConfig,
    GaussianLatencyConfig,
    OptimisticQueueConfig,
    ProbabilisticQueueConfig,
    RiskAverseQueueConfig,
    StaticFeeConfig,
    StaticLatencyConfig,
)
from gnomepy.java.backtest.orders import ExecutionReport
from gnomepy.java.backtest.runner import Backtest
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
    RiskConfig,
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
from gnomepy.entrypoint import run_backtest

__all__ = [
    # Top-level API
    "Strategy",
    "Backtest",
    "run_backtest",
    # Orders / OMS
    "Intent",
    "ExecutionReport",
    "OmsView",
    "PositionInfo",
    "TrackedOrderInfo",
    "RiskConfig",
    "BacktestResults",
    # Exchange config
    "ExchangeConfig",
    "StaticFeeConfig",
    "StaticLatencyConfig",
    "GaussianLatencyConfig",
    "OptimisticQueueConfig",
    "RiskAverseQueueConfig",
    "ProbabilisticQueueConfig",
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
]
