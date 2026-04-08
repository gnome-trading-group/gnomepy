from __future__ import annotations

from datetime import date, datetime

from gnomepy.java.backtest.config import ExchangeConfig
from gnomepy.java.backtest.runner import Backtest
from gnomepy.java.backtest.strategy import Strategy
from gnomepy.java.enums import SchemaType
from gnomepy.java.oms import RiskConfig
from gnomepy.java.recorder import BacktestResults


def run_backtest(
    strategy: Strategy | str,
    schema_type: SchemaType,
    start_date: date | datetime,
    end_date: date | datetime,
    exchanges: list[ExchangeConfig],
    *,
    bucket: str = "gnome-market-data-prod",
    s3_client=None,
    record: bool = True,
    risk_config: RiskConfig | None = None,
    progress: bool = True,
    strategy_args: dict | None = None,
    extra_jars: list[str] | None = None,
) -> BacktestResults | None:
    """Run a backtest end-to-end.

    `strategy` may be either:
      - a Python `gnomepy.Strategy` instance, or
      - a fully-qualified Java class name (e.g. ``"com.example.MyStrategy"``)
        that implements the Java ``BacktestStrategy`` interface.

    For Java strategies, ``strategy_args`` are passed via a
    ``configure(Map<String, Object>)`` method on the instance, and
    ``extra_jars`` is appended to the JVM classpath at startup.

    Returns `BacktestResults` if `record=True`, else None.
    """
    bt = Backtest(
        strategy=strategy,
        schema_type=schema_type,
        start_date=start_date,
        end_date=end_date,
        exchanges=exchanges,
        bucket=bucket,
        s3_client=s3_client,
        record=record,
        risk_config=risk_config,
        strategy_args=strategy_args,
        extra_jars=extra_jars,
    )
    return bt.run(progress=progress)
