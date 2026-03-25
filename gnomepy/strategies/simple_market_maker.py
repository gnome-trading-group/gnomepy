from __future__ import annotations

from gnomepy.java.backtest.strategy import Strategy
from gnomepy.java.backtest.orders import ExecutionReport
from gnomepy.java.oms import Intent
from gnomepy.java.schemas import JavaSchema
from gnomepy.signals.volatility_signal import VolatilitySignal


class SimpleMarketMaker(Strategy):
    """Simple market making strategy with volatility-based spread and inventory skew.

    Quotes bid/ask around mid price:
        bid = mid - half_spread + skew
        ask = mid + half_spread + skew

    Where:
        half_spread = max(vol_bps, min_spread_bps) * spread_multiplier * mid / 10000
        skew = -position * skew_bps_per_unit * mid / 10000

    Args:
        exchange_id: Exchange to trade on.
        security_id: Security to trade.
        vol_signal: Any VolatilitySignal (e.g., AdaptiveKalmanVolatility).
        spread_multiplier: Half-spread = vol × this.
        skew_bps_per_unit: Bps of skew per unit of inventory.
        order_size: Size per quote (scaled units).
        max_position: Pull one side when abs(position) exceeds this.
        min_spread_bps: Floor on half-spread.
    """

    def __init__(
        self,
        exchange_id: int,
        security_id: int,
        vol_signal: VolatilitySignal,
        spread_multiplier: float = 2.0,
        skew_bps_per_unit: float = 1.0,
        order_size: int = 1_000_000,
        max_position: int = 10_000_000,
        min_spread_bps: float = 1.0,
    ):
        self.exchange_id = exchange_id
        self.security_id = security_id
        self.vol_signal = vol_signal
        self.spread_multiplier = spread_multiplier
        self.skew_bps_per_unit = skew_bps_per_unit
        self.order_size = order_size
        self.max_position = max_position
        self.min_spread_bps = min_spread_bps
        self.fills: list[ExecutionReport] = []

    def on_market_data(self, timestamp: int, data: JavaSchema) -> list[Intent]:
        bid = data.bid_price(0)
        ask = data.ask_price(0)
        if bid <= 0 or ask <= 0:
            return []

        # Update vol signal
        self.vol_signal.update(timestamp, data)
        if not self.vol_signal.is_ready():
            return []

        mid = (bid + ask) // 2

        # Spread from volatility
        vol_bps = self.vol_signal.get_prediction()
        effective_vol = max(vol_bps, self.min_spread_bps)
        half_spread = effective_vol * self.spread_multiplier * mid / 10000

        # Inventory skew
        position = self.oms.get_position(self.exchange_id, self.security_id)
        net_qty = position.net_quantity if position else 0
        skew = -net_qty * self.skew_bps_per_unit * mid / 10000

        # Quote prices
        bid_price = int(mid - half_spread + skew)
        ask_price = int(mid + half_spread + skew)

        # Inventory limits — pull the side that increases exposure
        bid_size = self.order_size if net_qty < self.max_position else 0
        ask_size = self.order_size if net_qty > -self.max_position else 0

        return [Intent(
            exchange_id=self.exchange_id,
            security_id=self.security_id,
            bid_price=bid_price,
            bid_size=bid_size,
            ask_price=ask_price,
            ask_size=ask_size,
        )]

    def on_execution_report(self, timestamp: int, report: ExecutionReport) -> None:
        self.fills.append(report)
