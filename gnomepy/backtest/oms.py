from gnomepy.backtest.signal import Signal
from gnomepy.data.types import SchemaBase, Listing, Order, OrderType, TimeInForce, Intent, BasketIntent, OrderExecutionReport, ExecType
from gnomepy.backtest.signal import PositionAwareSignal

class SimpleOMS:

    def __init__(self, signals: list[Signal], notional: float):
        self.signals = signals
        self.notional = notional
        
        # Infer listings from signals
        all_listings = set()
        for signal in signals:
            all_listings.update(signal.listings)
        
        # Create listing_data ourselves - initialize as empty dict
        self.listing_data: dict[Listing, SchemaBase] = {}
        
        # Create signal_positions ourselves - initialize with empty positions for each signal
        self.signal_positions: dict[Signal, dict[Listing, float]] = {}
        for signal in signals:
            self.signal_positions[signal] = {listing: 0.0 for listing in signal.listings}
        
        # Create positions ourselves - initialize with zeros for all listings
        self.positions: dict[Listing, float] = {listing: 0.0 for listing in all_listings}

    def on_execution_report(self, execution_report: OrderExecutionReport):
        # Update positions based on the execution report
        listing = Listing(execution_report.exchange_id, execution_report.security_id)
        
        # Calculate position change based on filled quantity and side
        if execution_report.exec_type in [ExecType.FILL, ExecType.PARTIAL_FILL]:
            position_change = execution_report.filled_qty
            if execution_report.side == "sell":
                position_change = -position_change
            
            # Update overall positions
            if listing in self.positions:
                self.positions[listing] += position_change
            
            # Update signal-specific positions
            # Note: We need to track which signal generated the order to update the correct signal positions
            # For now, we'll update all signals that trade this listing proportionally
            signals_for_listing = [signal for signal in self.signals if listing in signal.listings]
            if signals_for_listing:
                position_change_per_signal = position_change / len(signals_for_listing)
                for signal in signals_for_listing:
                    self.signal_positions[signal][listing] += position_change_per_signal

        return
    
    def on_market_update(self, data: SchemaBase):
        # Update listing data history
        listing = Listing(data['exchange_id'], data["security_id"])
        if listing not in self.listing_data:
            self.listing_data[listing] = []
        self.listing_data[listing].append(data)

        # Generate intents from all signals
        all_intents = []
        for signal in self.signals:
            if isinstance(signal, PositionAwareSignal):
                new_intents = signal.process_new_tick(data=data, positions=self.signal_positions[signal])
            else:
                new_intents = signal.process_new_tick(data=data)
            all_intents.extend(new_intents)

        # Convert intents to orders
        orders = []
        for intent in all_intents:
            if isinstance(intent, BasketIntent):
                # Handle basket intent with proportions
                for sub_intent, proportion in zip(intent.intents, intent.proportions):
                    orders.append(self._create_order_from_intent(sub_intent, sub_intent.confidence * proportion))
            else:
                # Handle single intent
                orders.append(self._create_order_from_intent(intent, intent.confidence))
        
        return orders

    def _create_order_from_intent(self, intent: Intent, scaled_confidence: float) -> Order:
        """Create an order from an intent with scaled confidence."""
        order_size = int(self.notional * scaled_confidence / intent.listing.price)
        
        return Order(
            exchange_id=intent.listing.exchange_id,
            security_id=intent.listing.security_id,
            client_oid=None,
            price=intent.listing.price,  # Using current market price
            size=order_size,
            side=intent.side,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.IOC
        )