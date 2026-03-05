from gnomepy.data.types import Order, OrderType, TimeInForce, OrderExecutionReport, ExecType, FIXED_PRICE_SCALE, FIXED_SIZE_SCALE, CancelOrder
import logging

from gnomepy.research.oms.base import BaseOMS
from gnomepy.research.signals import Signal
from gnomepy.research.types import BasketIntent
from gnomepy.backtest.recorder import MarketRecorder

logger = logging.getLogger(__name__)


class MarketMakingOMS(BaseOMS):
    """OMS that supports limit orders for market making strategies.

    This OMS tracks active limit orders and cancels/replaces them when
    signal prices change. It's designed for market making strategies like
    Avellaneda-Stoikov that provide optimal bid/ask prices.
    """

    def __init__(
        self,
        signals: list[Signal],
        notional: float,
        starting_cash: float = 1000000.0,
        max_position_notional: float = None,
        position_aware_sizing: bool = True,
        position_scaling_factor: float = 0.5,
        tick_size: float = 0.01,
        passive_reprice_ticks: int = 3,
    ):
        """Initialize MarketMakingOMS with position-aware sizing controls.

        Parameters
        ----------
        signals : list[Signal]
            List of trading signals
        notional : float
            Base notional value for order sizing (in dollars).
            User supplies in dollars, used directly in calculations with midprice (also in dollars).
        starting_cash : float, default 1000000.0
            Starting cash balance
        max_position_notional : float, optional
            Maximum position notional value in dollars (absolute). If None, no limit.
            Stored in dollars to match current_position_notional calculation.
        position_aware_sizing : bool, default True
            If True, reduce order size as position grows
        position_scaling_factor : float, default 0.5
            Factor to reduce order size by when position grows (0.0-1.0)
            Lower values = more aggressive position reduction
        tick_size : float, default 0.01
            Minimum price increment in dollars. Desired prices are snapped to
            this grid before comparison with active orders.
        passive_reprice_ticks : int, default 3
            Number of ticks an order may passively drift before repricing.
            Preserves queue position for safe drift. 0 = always reprice
            immediately in both directions.
        """
        super().__init__(signals, notional, starting_cash)

        self.max_position_notional = max_position_notional
        self.position_aware_sizing = position_aware_sizing
        self.position_scaling_factor = position_scaling_factor
        if tick_size <= 0:
            raise ValueError("tick_size must be positive")
        self.tick_size = tick_size
        self.passive_reprice_ticks = passive_reprice_ticks

        # Track active orders: mapping of (listing_id, side) -> (client_oid, price)
        self.active_orders: dict[tuple[int, str], tuple[str, float]] = {}

        self._flatten_in_flight: set[int] = set()  # listing_ids with pending flatten orders

    def _calculate_position_aware_order_size(
        self,
        listing_id: int,
        side: str,
        base_notional: float,
        confidence: float,
        midprice: float,
    ) -> float:
        """Calculate order size with position-aware adjustments.

        Parameters
        ----------
        listing_id : int
            Listing ID for the order
        side : str
            Order side ("B" for buy, "A" for sell)
        base_notional : float
            Base notional value before adjustments (in dollars)
        confidence : float
            Confidence multiplier
        midprice : float
            Current mid price (in dollars, already scaled down from FIXED_PRICE_SCALE)

        Returns
        -------
        float
            Adjusted order size
        """
        base_size = abs(float(base_notional * confidence / midprice))

        if not self.position_aware_sizing:
            return base_size

        current_position = self.positions.get(listing_id, 0.0)
        current_position_notional = abs(current_position * midprice)

        # Check max position notional limit
        if self.max_position_notional is not None:
            if current_position_notional >= self.max_position_notional:
                if (side == "B" and current_position > 0) or (side == "A" and current_position < 0):
                    return 0.0

        # Calculate position-aware scaling
        if self.max_position_notional is not None and self.max_position_notional > 0:
            position_ratio = current_position_notional / self.max_position_notional
            scaling = 1.0 - (position_ratio * (1.0 - self.position_scaling_factor))
            scaling = max(scaling, self.position_scaling_factor)
        else:
            soft_limit = base_notional * 5.0
            if soft_limit > 0:
                position_ratio = min(current_position_notional / soft_limit, 1.0)
                scaling = 1.0 - (position_ratio * (1.0 - self.position_scaling_factor))
                scaling = max(scaling, self.position_scaling_factor)
            else:
                scaling = 1.0

        # Reduce size if adding to position in same direction
        if (side == "B" and current_position > 0) or (side == "A" and current_position < 0):
            scaling *= self.position_scaling_factor

        # Increase size if reducing position (opposite direction)
        elif (side == "B" and current_position < 0) or (side == "A" and current_position > 0):
            scaling *= 1.5
            scaling = min(scaling, 2.0)

        return base_size * scaling

    def _snap_to_tick(self, price: float) -> float:
        """Round price to nearest tick increment."""
        return round(price / self.tick_size) * self.tick_size

    def on_execution_report(self, timestamp: int, execution_report: OrderExecutionReport, recorder: MarketRecorder):
        """Handle execution reports from the exchange."""
        client_oid = execution_report.client_oid
        order = self.order_log.get(client_oid)
        if order is None:
            return

        listing_id = self._find_listing_id(execution_report.exchange_id, execution_report.security_id)
        if listing_id is None:
            return

        # Clear flatten-in-flight guard when flatten order completes
        if client_oid.startswith("flatten_"):
            if execution_report.exec_type in [ExecType.TRADE, ExecType.CANCELED, ExecType.REJECTED]:
                self._flatten_in_flight.discard(listing_id)

        # Remove from active orders if filled or cancelled
        if execution_report.exec_type in [ExecType.TRADE, ExecType.CANCELED]:
            order_key = (listing_id, order.side)
            if order_key in self.active_orders and self.active_orders[order_key][0] == client_oid:
                del self.active_orders[order_key]

        if execution_report.exec_type in [ExecType.TRADE] and execution_report.filled_qty > 0:
            self._apply_fill(listing_id, order, execution_report, timestamp, recorder)

        return

    def on_market_update(self, timestamp: int, market_update, market_recorder: MarketRecorder):
        """Process market update and generate/cancel orders as needed."""
        listing_id, all_intents = self._process_market_data(timestamp, market_update, market_recorder)
        if listing_id is None:
            return []

        if len(all_intents) == 0:
            return []

        # Handle flatten intents (circuit breaker) — generate market orders to close position
        flatten_orders = []
        for intent in all_intents:
            if isinstance(intent, BasketIntent):
                continue
            if not getattr(intent, 'flatten', False):
                continue
            listing_id_intent = intent.listing.listing_id
            if listing_id_intent in self._flatten_in_flight:
                continue
            current_position = self.positions.get(listing_id_intent, 0.0)
            if current_position == 0.0:
                continue
            side = "A" if current_position > 0 else "B"
            order_size = abs(current_position)
            listing = self._find_listing(listing_id_intent)
            if listing:
                self._order_counter += 1
                client_oid = f"flatten_{listing_id_intent}_{self._order_counter}"
                order = Order(
                    exchange_id=listing.exchange_id,
                    security_id=listing.security_id,
                    client_oid=client_oid,
                    price=None,
                    size=int(order_size * FIXED_SIZE_SCALE),
                    side=side,
                    order_type=OrderType.MARKET,
                    time_in_force=TimeInForce.GTC,
                )
                self.order_log[client_oid] = order
                self._flatten_in_flight.add(listing_id_intent)
                flatten_orders.append(order)

        # Convert intents to limit orders and manage active orders
        orders = []
        cancel_orders = []

        # Track which orders we want to keep/update
        desired_orders: dict[tuple[int, str], tuple[float, float]] = {}

        for intent in all_intents:
            if isinstance(intent, BasketIntent):
                for sub_intent, proportion in zip(intent.intents, intent.proportions):
                    if sub_intent.price is not None:
                        listing_id_intent = sub_intent.listing.listing_id
                        normalized_side = "A" if sub_intent.side in ["S", "A"] else "B"
                        order_key = (listing_id_intent, normalized_side)

                        if listing_id_intent in self.listing_data:
                            bid_prices = self.listing_data[listing_id_intent]['bidPrice0']
                            ask_prices = self.listing_data[listing_id_intent]['askPrice0']
                            midprice = (bid_prices[-1] + ask_prices[-1]) / 2
                            order_size = self._calculate_position_aware_order_size(
                                listing_id=listing_id_intent,
                                side=normalized_side,
                                base_notional=self.notional * proportion,
                                confidence=sub_intent.confidence,
                                midprice=midprice,
                            )
                        else:
                            continue

                        if order_size > 0:
                            desired_orders[order_key] = (sub_intent.price, order_size)
            else:
                if intent.price is not None:
                    listing_id_intent = intent.listing.listing_id
                    normalized_side = "A" if intent.side in ["S", "A"] else "B"
                    order_key = (listing_id_intent, normalized_side)

                    if listing_id_intent in self.listing_data:
                        bid_prices = self.listing_data[listing_id_intent]['bidPrice0']
                        ask_prices = self.listing_data[listing_id_intent]['askPrice0']
                        midprice = (bid_prices[-1] + ask_prices[-1]) / 2
                        order_size = self._calculate_position_aware_order_size(
                            listing_id=listing_id_intent,
                            side=normalized_side,
                            base_notional=self.notional,
                            confidence=intent.confidence,
                            midprice=midprice,
                        )
                    else:
                        continue

                    if order_size > 0:
                        desired_orders[order_key] = (intent.price, order_size)

        # --- Reconcile active orders vs desired orders ---

        # 1. Cancel orders whose intent is gone entirely
        for order_key, (existing_oid, existing_price) in list(self.active_orders.items()):
            if order_key not in desired_orders:
                listing = self._find_listing(order_key[0])
                if listing:
                    cancel_orders.append(CancelOrder(
                        exchange_id=listing.exchange_id,
                        security_id=listing.security_id,
                        client_oid=existing_oid,
                    ))
                    del self.active_orders[order_key]

        # 2. For each desired order: snap to tick grid, apply asymmetric urgency
        for order_key, (desired_price, desired_size) in desired_orders.items():
            listing_id_key, side = order_key
            snapped_price = self._snap_to_tick(desired_price)

            if order_key in self.active_orders:
                existing_oid, existing_price = self.active_orders[order_key]
                ticks_away = round(abs(snapped_price - existing_price) / self.tick_size)

                if ticks_away == 0:
                    continue

                # Asymmetric urgency: is existing order too aggressive?
                if side == "B":
                    too_aggressive = existing_price > snapped_price
                else:
                    too_aggressive = existing_price < snapped_price

                if not too_aggressive and ticks_away <= self.passive_reprice_ticks:
                    continue

                # Cancel existing order (aggressive, or passive beyond tolerance)
                listing = self._find_listing(listing_id_key)
                if listing:
                    cancel_orders.append(CancelOrder(
                        exchange_id=listing.exchange_id,
                        security_id=listing.security_id,
                        client_oid=existing_oid,
                    ))
                    del self.active_orders[order_key]

            # Create new order (either no active order, or just cancelled one above)
            if order_key not in self.active_orders:
                listing = self._find_listing(listing_id_key)
                if listing:
                    self._order_counter += 1
                    client_oid = f"limit_oms_{self._order_counter}_{listing_id_key}_{side}"
                    order = Order(
                        exchange_id=listing.exchange_id,
                        security_id=listing.security_id,
                        client_oid=client_oid,
                        price=int(snapped_price * FIXED_PRICE_SCALE),
                        size=int(desired_size * FIXED_SIZE_SCALE),
                        side=side,
                        order_type=OrderType.LIMIT,
                        time_in_force=TimeInForce.GTC,
                    )
                    self.order_log[client_oid] = order
                    self.active_orders[order_key] = (client_oid, snapped_price)
                    orders.append(order)

        return cancel_orders + flatten_orders + orders
