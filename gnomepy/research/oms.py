from gnomepy.data.types import SchemaBase, Order, OrderType, TimeInForce, OrderExecutionReport, ExecType, FIXED_PRICE_SCALE, FIXED_SIZE_SCALE
import dataclasses
import time
import logging
import numpy as np

from gnomepy.research.signal import Signal, PositionAwareSignal
from gnomepy.research.types import BasketIntent, Intent
from gnomepy.backtest.recorder import Recorder, RecordType

# Set up logger for performance tracking
logger = logging.getLogger(__name__)


class SimpleOMS:

    def __init__(self, signals: list[Signal], notional: float, starting_cash: float = 1000000.0):
        self.signals = signals
        self.notional = notional
        self.cash = starting_cash
        
        # Infer listings from signals
        all_listings = []
        for signal in signals:
            all_listings.extend(signal.listings)
        
        # Create listing_data using numpy arrays - initialize as empty dict using listing IDs as keys
        self.listing_data: dict[int, dict[str, np.ndarray]] = {}
        
        # Create signal_positions ourselves - initialize with empty positions for each signal
        self.signal_positions: dict[Signal, dict[int, float]] = {}
        for signal in signals:
            self.signal_positions[signal] = {listing.listing_id: 0.0 for listing in signal.listings}
        
        # Create positions ourselves - initialize with zeros for all listings using listing IDs as keys
        self.positions: dict[int, float] = {listing.listing_id: 0.0 for listing in all_listings}
        # Add order log to keep history of all submitted orders
        self.order_log: dict[str, Order] = {}
        
        # Track elapsed ticks for each listing to control data appending frequency
        self.elapsed_ticks: dict[int, int] = {listing.listing_id: 0 for listing in all_listings}

    def on_execution_report(self, timestamp: int, execution_report: OrderExecutionReport, recorder: Recorder):
        client_oid = execution_report.client_oid
        order = self.order_log.get(client_oid)
        if order is None:
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

        if execution_report.exec_type in [ExecType.TRADE]:
            # Use order.side to determine position change direction
            filled_qty = execution_report.filled_qty
            filled_price = execution_report.filled_price / FIXED_PRICE_SCALE  # Scale the price
            position_change = filled_qty if order.side == "B" else -filled_qty

            # Update cash based on the trade
            trade_value = filled_qty * filled_price
            previous_cash = self.cash
            if order.side == "B":
                # Buying - cash decreases
                self.cash -= trade_value
                print(f"Cash update - BUY: {previous_cash:.2f} -> {self.cash:.2f} (trade value: {trade_value:.2f})")
            else:
                # Selling - cash increases
                self.cash += trade_value
                print(f"Cash update - SELL: {previous_cash:.2f} -> {self.cash:.2f} (trade value: {trade_value:.2f})")

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

            recorder.log(
                event=RecordType.EXECUTION,
                listing_id=listing_id,
                timestamp=timestamp,
                price=filled_price,
                quantity=self.positions[listing_id],
                fee=execution_report.fee / FIXED_PRICE_SCALE,
            )

        return
    
    def on_market_update(self, timestamp: int, market_update: SchemaBase, recorder: Recorder):
        start_time = time.perf_counter()
        
        # Update listing data history using listing_id as key
        listing_id = None
        
        # Find the listing_id based on exchange_id and security_id
        find_listing_start = time.perf_counter()
        for signal in self.signals:
            for listing in signal.listings:
                if (listing.exchange_id == market_update.exchange_id and 
                    listing.security_id == market_update.security_id):
                    listing_id = listing.listing_id
                    break
            if listing_id:
                break
        find_listing_time = time.perf_counter() - find_listing_start
        # print(f"Find listing ID: {find_listing_time:.6f}s")
        
        if listing_id is None:
            return []  # Unknown listing, skip
        
        # Increment elapsed ticks for this listing
        self.elapsed_ticks[listing_id] += 1
        
        # Determine if we should append this data based on trade frequency
        # Get the minimum trade frequency across all signals
        min_trade_frequency = min(
            getattr(signal, 'trade_frequency', 1) for signal in self.signals
        )
        
        # Only append data if we've reached the trade frequency threshold
        should_append_data = (self.elapsed_ticks[listing_id] % min_trade_frequency == 0)
        # print(f"Listing {listing_id}: tick {self.elapsed_ticks[listing_id]}, trade_freq {min_trade_frequency}, append: {should_append_data}")
        
        # Initialize numpy arrays if needed
        init_arrays_start = time.perf_counter()
        if listing_id not in self.listing_data:
            # Initialize empty numpy arrays for this listing
            self.listing_data[listing_id] = {}
        init_arrays_time = time.perf_counter() - init_arrays_start
        # print(f"Initialize arrays: {init_arrays_time:.6f}s")
        
        # Convert market data to dict and flatten levels
        convert_data_start = time.perf_counter()
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

        recorder.log_market_event(
            listing_id=listing_id,
            timestamp=timestamp,
            market_update=market_update,
            quantity=self.positions[listing_id],
        )

        convert_data_time = time.perf_counter() - convert_data_start
        # print(f"Convert market data: {convert_data_time:.6f}s")

        # Add new data to numpy arrays (much faster than pandas) - only if we should append data
        update_arrays_start = time.perf_counter()
        if should_append_data:
            max_history_records = max([self.signals[i].max_lookback for i in range(len(self.signals))]) # Configurable parameter
            
            for column, value in market_dict.items():
                # Skip non-numeric fields that can't be converted to float
                if isinstance(value, str):
                    continue
                
                # Skip any None values
                if value is None:
                    continue
                
                try:
                    # Try to convert to float to ensure it's numeric
                    float_value = float(value)
                except (ValueError, TypeError):
                    # Skip fields that can't be converted to float
                    continue
                    
                if column not in self.listing_data[listing_id]:
                    # Initialize new column array
                    self.listing_data[listing_id][column] = np.array([float_value], dtype=np.float64)
                else:
                    # Append to existing array
                    current_array = self.listing_data[listing_id][column]
                    new_array = np.append(current_array, float_value)
                    
                    # Keep only the last N records
                    if len(new_array) > max_history_records:
                        new_array = new_array[-max_history_records:]
                    
                    self.listing_data[listing_id][column] = new_array
        update_arrays_time = time.perf_counter() - update_arrays_start
        # print(f"Update numpy arrays: {update_arrays_time:.6f}s (appended: {should_append_data})")

        # Generate intents from all signals
        generate_intents_start = time.perf_counter()
        all_intents = []
        
        for signal in self.signals:
            if isinstance(signal, PositionAwareSignal):
                new_intents = signal.process_new_tick(data=self.listing_data, positions=self.signal_positions[signal], ticker_listing_id=listing_id)
            else:
                new_intents = signal.process_new_tick(data=self.listing_data)
            
            if new_intents and len(new_intents) > 0:
                all_intents.extend(new_intents)
        generate_intents_time = time.perf_counter() - generate_intents_start
        # print(f"Generate intents: {generate_intents_time:.6f}s")

        # Convert intents to orders
        convert_orders_start = time.perf_counter()
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
        convert_orders_time = time.perf_counter() - convert_orders_start
        # print(f"Convert intents to orders: {convert_orders_time:.6f}s")
        
        total_time = time.perf_counter() - start_time
        # print(f"Total on_market_update time: {total_time:.6f}s")
        
        return orders

    def _create_order_from_intent(self, intent: Intent, scaled_confidence: float) -> Order:
        """Create an order from an intent with scaled confidence, or flatten position if requested."""
        listing_id = intent.listing.listing_id
        # Get latest data from numpy arrays
        bid_prices = self.listing_data[listing_id]['bidPrice0']
        ask_prices = self.listing_data[listing_id]['askPrice0']
        latest_bid = bid_prices[-1]
        latest_ask = ask_prices[-1]
        midprice = (latest_bid + latest_ask) / 2

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

        order = Order(
            exchange_id=intent.listing.exchange_id,
            security_id=intent.listing.security_id,
            client_oid=None,
            price=None,  # There is no price for Market Orders
            size=order_size,
            side=side,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC
        )
        
        return order