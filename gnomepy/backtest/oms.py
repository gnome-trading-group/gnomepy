from gnomepy.backtest.signal import Signal
from gnomepy.data.types import SchemaBase, Listing, Order, OrderType, TimeInForce, Intent, BasketIntent, OrderExecutionReport, ExecType, FIXED_PRICE_SCALE, FIXED_SIZE_SCALE
from gnomepy.backtest.signal import PositionAwareSignal
from gnomepy.data.common import DataStore
import pandas as pd
import dataclasses
import time

class SimpleOMS:

    def __init__(self, signals: list[Signal], notional: float):
        self.signals = signals
        self.notional = notional
        
        # Infer listings from signals
        all_listings = []
        for signal in signals:
            all_listings.extend(signal.listings)
        
        # Create listing_data ourselves - initialize as empty dict using listing IDs as keys
        self.listing_data: dict[int, DataStore] = {}
        
        # Create signal_positions ourselves - initialize with empty positions for each signal
        self.signal_positions: dict[Signal, dict[int, float]] = {}
        for signal in signals:
            self.signal_positions[signal] = {listing.listing_id: 0.0 for listing in signal.listings}
        
        # Create positions ourselves - initialize with zeros for all listings using listing IDs as keys
        self.positions: dict[int, float] = {listing.listing_id: 0.0 for listing in all_listings}
        # Add order log to keep history of all submitted orders
        self.order_log: dict[str, Order] = {}

    def on_execution_report(self, execution_report: OrderExecutionReport):
        client_oid = execution_report.client_oid
        order = self.order_log.get(client_oid)
        if order is None:
            print(f"Unknown order for OID {client_oid}, skipping position update.")
            return

        listing_id = None
        for signal in self.signals:
            for listing in signal.listings:
                if (listing.exchange_id == execution_report.exchange_id and 
                    listing.security_id == execution_report.security_id):
                    listing_id = listing.listing_id
                    break
            if listing_id:
                break
        if listing_id is None:
            return  # Unknown listing, skip

        if execution_report.exec_type in [ExecType.FILL, ExecType.PARTIAL_FILL]:
            # Use order.side to determine position change direction
            filled_qty = execution_report.filled_qty
            position_change = filled_qty if order.side == "B" else -filled_qty

            # Update overall positions
            if listing_id in self.positions:
                self.positions[listing_id] += position_change

            # Update signal-specific positions
            signals_for_listing = [signal for signal in self.signals 
                                 if any(listing.listing_id == listing_id for listing in signal.listings)]
            if signals_for_listing:
                position_change_per_signal = position_change / len(signals_for_listing)
                for signal in signals_for_listing:
                    if listing_id in self.signal_positions[signal]:
                        self.signal_positions[signal][listing_id] += position_change_per_signal

            print(f"Position update for listing {listing_id}: change={position_change}")
            print(f"Updated positions: {self.positions}")
        return
    
    def on_market_update(self, market_update: SchemaBase):
        # Update listing data history using listing_id as key
        listing_id = None
        
        # Find the listing_id based on exchange_id and security_id
        for signal in self.signals:
            for listing in signal.listings:
                if (listing.exchange_id == market_update.exchange_id and 
                    listing.security_id == market_update.security_id):
                    listing_id = listing.listing_id
                    break
            if listing_id:
                break
        
        if listing_id is None:
            return []  # Unknown listing, skip
        
        # Update the running history for this listing
        if listing_id not in self.listing_data:
            # Initialize empty DataFrame for this listing
            self.listing_data[listing_id] = pd.DataFrame()
        
        # Add new market update to history
        # Convert market data to dict and flatten levels
        market_dict = dataclasses.asdict(market_update)
        levels = market_dict.pop('levels', [])
        # Add flattened level data
        for i, level in enumerate(levels):
            # Manually scale price and size fields
            market_dict[f'bidPrice{i}'] = level.get('bid_px', 0) / FIXED_PRICE_SCALE
            market_dict[f'askPrice{i}'] = level.get('ask_px', 0) / FIXED_PRICE_SCALE
            market_dict[f'bidSize{i}'] = level.get('bid_sz', 0) / FIXED_SIZE_SCALE
            market_dict[f'askSize{i}'] = level.get('ask_sz', 0) / FIXED_SIZE_SCALE
            market_dict[f'bidCount{i}'] = level.get('bid_ct', 0)
            market_dict[f'askCount{i}'] = level.get('ask_ct', 0)

        new_row = pd.DataFrame([market_dict])
        self.listing_data[listing_id] = pd.concat([self.listing_data[listing_id], new_row], ignore_index=True)

        # Keep only the last N records (adjust as needed)
        max_history_records = 1000  # Configurable parameter
        self.listing_data[listing_id] = self.listing_data[listing_id].tail(max_history_records)

        # Generate intents from all signals
        all_intents = []
        for signal in self.signals:
            if isinstance(signal, PositionAwareSignal):
                new_intents = signal.process_new_tick(data=self.listing_data, positions=self.signal_positions[signal], ticker_listing_id=listing_id)
            else:
                new_intents = signal.process_new_tick(data=self.listing_data)

            if new_intents and len(new_intents) > 0:
                all_intents.extend(new_intents)

        # Convert intents to orders
        orders = []
        for intent in all_intents:
            if isinstance(intent, BasketIntent):
                for sub_intent, proportion in zip(intent.intents, intent.proportions):
                    order = self._create_order_from_intent(sub_intent, sub_intent.confidence * proportion)
                    if order is not None:
                        # Assign a client_oid if not already set
                        if order.client_oid is None:
                            order.client_oid = f"oms_{int(time.time() * 1e9)}"
                        self.order_log[order.client_oid] = order
                        orders.append(order)
            else:
                order = self._create_order_from_intent(intent, intent.confidence)
                if order is not None:
                    if order.client_oid is None:
                        order.client_oid = f"oms_{int(time.time() * 1e9)}"
                    self.order_log[order.client_oid] = order
                    orders.append(order)
        return orders

    def _create_order_from_intent(self, intent: Intent, scaled_confidence: float) -> Order:
        """Create an order from an intent with scaled confidence, or flatten position if requested."""
        listing_id = intent.listing.listing_id
        latest_data = self.listing_data[listing_id].iloc[-1]
        midprice = (latest_data['bidPrice0'] + latest_data['askPrice0']) / 2

        if getattr(intent, "flatten", False):
            # Generate order to flatten position
            current_position = self.positions[listing_id]
            
            # Determine side and size based on current position
            if current_position > 0:
                # We have a long position, need to sell to flatten
                side = "S"
                order_size = abs(current_position)
            elif current_position < 0:
                # We have a short position, need to buy to flatten
                side = "B"
                order_size = abs(current_position)
            else:
                # No position to flatten
                return None
        else:
            order_size = abs(float(self.notional * scaled_confidence / midprice))
            side = intent.side

        return Order(
            exchange_id=intent.listing.exchange_id,
            security_id=intent.listing.security_id,
            client_oid=None,
            price=None,  # There is no price for Market Orders
            size=order_size,
            side=side,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC
        )