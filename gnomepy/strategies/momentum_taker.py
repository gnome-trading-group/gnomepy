from __future__ import annotations

from gnomepy.java.backtest.strategy import Strategy
from gnomepy.java.backtest.orders import ExecutionReport
from gnomepy.java.enums import Side, OrderType
from gnomepy.java.oms import Intent
from gnomepy.java.schemas import JavaSchema


class MomentumTaker(Strategy):
    """Simple momentum-following strategy for backtest testing.

    Tracks mid price over a lookback window. When momentum exceeds
    a threshold (in bps), submits an aggressive take intent
    in the direction of the move.

    Args:
        exchange_id: Exchange to trade on.
        security_id: Security to trade.
        lookback: Number of ticks for momentum measurement.
        threshold_bps: Minimum momentum (bps) to trigger a trade.
        order_size: Size per order (scaled units).
        cooldown_ticks: Minimum ticks between trades.
    """

    def __init__(
        self,
        exchange_id: int,
        security_id: int,
        lookback: int = 100,
        threshold_bps: float = 5.0,
        order_size: int = 1_000_000,
        cooldown_ticks: int = 50,
    ):
        self.exchange_id = exchange_id
        self.security_id = security_id
        self.lookback = lookback
        self.threshold_bps = threshold_bps
        self.order_size = order_size
        self.cooldown_ticks = cooldown_ticks

        self.mid_history: list[int] = []
        self.ticks_since_trade = 0
        self.fills: list[ExecutionReport] = []

    def on_market_data(self, timestamp: int, data: JavaSchema) -> list[Intent]:
        bid = data.bid_price(0)
        ask = data.ask_price(0)
        if bid <= 0 or ask <= 0:
            return []

        mid = (bid + ask) // 2
        self.mid_history.append(mid)
        self.ticks_since_trade += 1

        if len(self.mid_history) < self.lookback:
            return []
        if len(self.mid_history) > self.lookback:
            self.mid_history.pop(0)

        if self.ticks_since_trade < self.cooldown_ticks:
            return []

        old_mid = self.mid_history[0]
        if old_mid == 0:
            return []

        momentum_bps = (mid - old_mid) / old_mid * 10000

        if momentum_bps > self.threshold_bps:
            self.ticks_since_trade = 0
            return [Intent(
                exchange_id=self.exchange_id,
                security_id=self.security_id,
                take_side=Side.BID,
                take_size=self.order_size,
                take_order_type=OrderType.MARKET,
            )]
        elif momentum_bps < -self.threshold_bps:
            self.ticks_since_trade = 0
            return [Intent(
                exchange_id=self.exchange_id,
                security_id=self.security_id,
                take_side=Side.ASK,
                take_size=self.order_size,
                take_order_type=OrderType.MARKET,
            )]

        return []

    def on_execution_report(self, timestamp: int, report: ExecutionReport) -> None:
        self.fills.append(report)
