from __future__ import annotations

from gnomepy.java.backtest.strategy import Strategy
from gnomepy.java.backtest.orders import ExecutionReport
from gnomepy.java.enums import Side, OrderType
from gnomepy.java.oms import Intent
from gnomepy.java.schemas import JavaSchema
from gnomepy.signals.fair_value import FairValueModel, MidFairValue
from gnomepy.signals.fill_rate_signal import FillRateSignal
from gnomepy.signals.volatility import VolatilitySignal


class SimpleMarketMaker(Strategy):
    """Market making strategy with pluggable fair value, asymmetric spread skew,
    adaptive fill rate, size taper, and active inventory flattening.

    Quotes bid/ask around fair value with asymmetric spread:
        bid = fair_value - half_spread * (1 + skew_factor)
        ask = fair_value + half_spread * (1 - skew_factor)

    When long: bid widens (fewer buys), ask tightens (more sells).
    When short: opposite. Both sides maintain positive edge relative to fair value.

    Args:
        security_id: Security to trade.
        exchange_id: Exchange to trade on.
        vol_signal: Any VolatilitySignal (e.g., SpreadVolatility).
        fair_value_model: Fair value estimator. Default: MidFairValue (simple mid).
        spread_multiplier: Base half-spread = vol × this.
        skew_intensity: How asymmetric the spread becomes with inventory (0.0-1.0).
            0.0 = symmetric. 0.5 = reducing side 50% tighter at max inventory.
            1.0 = reducing side at fair value at max inventory.
        order_size: Size per quote (scaled units).
        max_position: Position limit (scaled).
        min_spread_bps: Floor on half-spread.
        flatten_threshold: Fraction of max_position that triggers active flattening (0.0-1.0).
        target_fill_rate: Desired fills per minute for adaptive spread.
        fill_rate_window_sec: Rolling window for fill rate measurement.
        min_spread_scale: Minimum fill-rate spread adjustment.
        max_spread_scale: Maximum fill-rate spread adjustment.
    """

    def __init__(
        self,
        security_id: int,
        exchange_id: int,
        vol_signal: VolatilitySignal,
        fair_value_model: FairValueModel | None = None,
        spread_multiplier: float = 2.0,
        skew_intensity: float = 0.5,
        order_size: int = 1_000_000,
        max_position: int = 10_000_000,
        min_spread_bps: float = 1.0,
        flatten_threshold: float = 1.0,
        target_fill_rate: float = 20.0,
        fill_rate_window_sec: float = 60.0,
        min_spread_scale: float = 0.5,
        max_spread_scale: float = 3.0,
    ):
        self.security_id = security_id
        self.exchange_id = exchange_id
        self.vol_signal = vol_signal
        self.fair_value_model = fair_value_model or MidFairValue()
        self.spread_multiplier = spread_multiplier
        self.skew_intensity = skew_intensity
        self.order_size = order_size
        self.max_position = max_position
        self.min_spread_bps = min_spread_bps
        self.flatten_threshold = flatten_threshold
        self.fill_rate_signal = FillRateSignal(
            target_fill_rate=target_fill_rate,
            window_sec=fill_rate_window_sec,
            min_scale=min_spread_scale,
            max_scale=max_spread_scale,
        )
        self.fills: list[ExecutionReport] = []

    def on_market_data(self, timestamp: int, data: JavaSchema) -> list[Intent]:
        bid = data.bid_price(0)
        ask = data.ask_price(0)
        if bid <= 0 or ask <= 0:
            return [self._cancel_intent()]

        # Update signals
        self.vol_signal.update(timestamp, data)
        self.fair_value_model.update(timestamp, data)
        if not self.vol_signal.is_ready() or not self.fair_value_model.is_ready():
            return [self._cancel_intent()]

        fair_value = self.fair_value_model.get_fair_value()

        position = self.oms.get_position(self.exchange_id, self.security_id)
        confirmed = position.net_quantity if position else 0
        inventory_ratio = confirmed / self.max_position

        # Spread from volatility, adapted by fill rate
        vol_bps = self.vol_signal.get_prediction()
        effective_vol = max(vol_bps, self.min_spread_bps)
        fill_adj = self.fill_rate_signal.get_spread_adjustment(timestamp)
        half_spread = effective_vol * self.spread_multiplier * fill_adj * fair_value / 10000

        # Asymmetric spread skew — tighter on reducing side, wider on risky side
        # skew_factor > 0 when long: bid wider (fewer buys), ask tighter (more sells)
        # skew_factor < 0 when short: ask wider (fewer sells), bid tighter (more buys)
        clamped_ratio = max(-1.0, min(1.0, inventory_ratio))
        skew_factor = clamped_ratio * self.skew_intensity
        bid_half = half_spread * (1 + skew_factor)
        ask_half = max(1, half_spread * (1 - skew_factor))

        # Quote prices
        raw_bid = int(fair_value - bid_half)
        raw_ask = int(fair_value + ask_half)
        abs_ratio = min(abs(inventory_ratio), 1.0)

        if inventory_ratio > 0:
            # Long — bid is risky (widen), ask is reducing (tighten toward best ask)
            bid_price = min(raw_bid, bid)
            # Blend ask between model price and best ask based on inventory urgency
            ask_price = int(raw_ask * (1 - abs_ratio) + ask * abs_ratio)
        elif inventory_ratio < 0:
            # Short — ask is risky (widen), bid is reducing (tighten toward best bid)
            ask_price = max(raw_ask, ask)
            # Blend bid between model price and best bid based on inventory urgency
            bid_price = int(raw_bid * (1 - abs_ratio) + bid * abs_ratio)
        else:
            bid_price = min(raw_bid, bid)
            ask_price = max(raw_ask, ask)

        # Linear size taper — risky side shrinks as inventory grows
        if inventory_ratio >= 1.0:
            bid_size = 0
        else:
            bid_size = int(self.order_size * max(0.0, 1.0 - max(0.0, inventory_ratio)))

        if inventory_ratio <= -1.0:
            ask_size = 0
        else:
            ask_size = int(self.order_size * max(0.0, 1.0 - max(0.0, -inventory_ratio)))

        # Active flattening
        flatten_level = int(self.flatten_threshold * self.max_position)

        take_side = None
        take_size = 0
        take_order_type = None
        take_limit_price = 0

        if confirmed > flatten_level:
            take_side = Side.ASK
            take_size = confirmed# - flatten_level
            take_order_type = OrderType.MARKET
            bid_size = 0
        elif confirmed < -flatten_level:
            take_side = Side.BID
            take_size = abs(confirmed)# - flatten_level
            take_order_type = OrderType.MARKET
            ask_size = 0

        if take_side is not None:
            return [Intent(
                exchange_id=self.exchange_id,
                security_id=self.security_id,
                bid_price=bid_price,
                bid_size=bid_size,
                ask_price=ask_price,
                ask_size=ask_size,
                take_side=take_side,
                take_size=take_size,
                take_order_type=take_order_type,
                take_limit_price=take_limit_price,
            )]

        return [Intent(
            exchange_id=self.exchange_id,
            security_id=self.security_id,
            bid_price=bid_price,
            bid_size=bid_size,
            ask_price=ask_price,
            ask_size=ask_size,
        )]

    def _cancel_intent(self) -> Intent:
        """Intent with zero sizes — tells OMS to cancel any resting orders."""
        return Intent(exchange_id=self.exchange_id, security_id=self.security_id)

    def on_execution_report(self, timestamp: int, report: ExecutionReport) -> None:
        self.fills.append(report)
        if report.exec_type.value in ("FILL", "PARTIAL_FILL"):
            self.fill_rate_signal.on_fill(timestamp)
