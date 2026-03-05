from gnomepy.data.types import Order, OrderType, TimeInForce, OrderExecutionReport, ExecType, FIXED_SIZE_SCALE
import logging

from gnomepy.research.oms.base import BaseOMS
from gnomepy.research.signals import Signal
from gnomepy.research.types import BasketIntent, Intent
from gnomepy.backtest.recorder import MarketRecorder

logger = logging.getLogger(__name__)


class SimpleOMS(BaseOMS):

    def __init__(self, signals: list[Signal], notional: float, starting_cash: float = 1000000.0):
        super().__init__(signals, notional, starting_cash)

    def on_execution_report(self, timestamp: int, execution_report: OrderExecutionReport, recorder: MarketRecorder):
        client_oid = execution_report.client_oid
        order = self.order_log.get(client_oid)
        if order is None:
            return

        listing_id = self._find_listing_id(execution_report.exchange_id, execution_report.security_id)
        if listing_id is None:
            return

        if execution_report.exec_type == ExecType.REJECTED:
            signals_for_listing = [signal for signal in self.signals
                                 if any(listing.listing_id == listing_id for listing in signal.listings)]
            for signal in signals_for_listing:
                if hasattr(signal, '_entry_pending'):
                    signal._entry_pending = False
                if hasattr(signal, '_exit_pending'):
                    signal._exit_pending = False
            return

        if execution_report.exec_type in [ExecType.TRADE] and execution_report.filled_qty > 0:
            self._apply_fill(listing_id, order, execution_report, timestamp, recorder)

        return

    def on_market_update(self, timestamp: int, market_update, market_recorder: MarketRecorder):
        listing_id, all_intents = self._process_market_data(timestamp, market_update, market_recorder)
        if listing_id is None:
            return []

        # Convert intents to orders
        orders = []
        for intent in all_intents:
            if isinstance(intent, BasketIntent):
                for sub_intent, proportion in zip(intent.intents, intent.proportions):
                    order = self._create_order_from_intent(sub_intent, sub_intent.confidence * proportion)
                    if order is not None:
                        if order.client_oid is None:
                            order.client_oid = self._next_client_oid()
                        self.order_log[order.client_oid] = order
                        orders.append(order)
            else:
                order = self._create_order_from_intent(intent, intent.confidence)
                if order is not None:
                    if order.client_oid is None:
                        order.client_oid = self._next_client_oid()
                    self.order_log[order.client_oid] = order
                    orders.append(order)
                else:
                    # Order creation failed — clear pending flags so signal isn't stuck
                    intent_listing_id = intent.listing.listing_id
                    signals_for_listing = [s for s in self.signals
                                         if any(l.listing_id == intent_listing_id for l in s.listings)]
                    for s in signals_for_listing:
                        if hasattr(s, '_entry_pending'):
                            s._entry_pending = False
                        if hasattr(s, '_exit_pending'):
                            s._exit_pending = False

        return orders

    def _create_order_from_intent(self, intent: Intent, scaled_confidence: float) -> Order:
        """Create an order from an intent with scaled confidence, or flatten position if requested."""
        listing_id = intent.listing.listing_id
        bid_prices = self.listing_data[listing_id]['bidPrice0']
        ask_prices = self.listing_data[listing_id]['askPrice0']
        midprice = (bid_prices[-1] + ask_prices[-1]) / 2

        if midprice <= 0:
            return None

        if getattr(intent, "flatten", False):
            current_position = self.positions[listing_id]
            if current_position > 0:
                side = "S"
                order_size = abs(current_position)
            elif current_position < 0:
                side = "B"
                order_size = abs(current_position)
            else:
                return None
        else:
            order_size = abs(float(self.notional * scaled_confidence / midprice))
            side = intent.side

        order = Order(
            exchange_id=intent.listing.exchange_id,
            security_id=intent.listing.security_id,
            client_oid=None,
            price=None,
            size=order_size * FIXED_SIZE_SCALE,
            side=side,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC
        )

        return order
