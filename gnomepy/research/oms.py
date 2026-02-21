from gnomepy.data.types import SchemaBase, Order, OrderType, TimeInForce, OrderExecutionReport, ExecType, FIXED_PRICE_SCALE, FIXED_SIZE_SCALE, CancelOrder, CancelOrder
import dataclasses
import time
import logging
import numpy as np

from gnomepy.research.signals import Signal, PositionAwareSignal
from gnomepy.research.types import BasketIntent, Intent
from gnomepy.backtest.recorder import MarketRecorder, RecordType

logger = logging.getLogger(__name__)


class SimpleOMS:

    def __init__(self, signals: list[Signal], notional: float, starting_cash: float = 1000000.0):
        # notional: user supplies in dollars, used as-is for order sizing calculations
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
        self._order_counter: int = 0

        # Track elapsed ticks for each listing to control data appending frequency
        self.elapsed_ticks: dict[int, int] = {listing.listing_id: 0 for listing in all_listings}

    def _next_client_oid(self) -> str:
        self._order_counter += 1
        return f"oms_{self._order_counter}"

    def on_execution_report(self, timestamp: int, execution_report: OrderExecutionReport, recorder: MarketRecorder):
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
            # Use order.side to determine position change direction
            filled_qty = execution_report.filled_qty / FIXED_SIZE_SCALE
            mid_price = execution_report.mid_price / FIXED_PRICE_SCALE  # Scale the price
            filled_price = execution_report.filled_price / FIXED_PRICE_SCALE  # Scale the price
            fee = execution_report.fee / (FIXED_PRICE_SCALE * FIXED_SIZE_SCALE)
            position_change = filled_qty if order.side == "B" else -filled_qty

            # Update cash based on the trade
            trade_value = filled_qty * filled_price
            previous_cash = self.cash
            if order.side == "B":
                # Buying - cash decreases
                self.cash -= trade_value
                # print(f"Cash update - BUY: {previous_cash:.2f} -> {self.cash:.2f} (trade value: {trade_value:.2f})")
            else:
                # Selling - cash increases
                self.cash += trade_value
                # print(f"Cash update - SELL: {previous_cash:.2f} -> {self.cash:.2f} (trade value: {trade_value:.2f})")

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
                price=mid_price if mid_price > 0 else filled_price,
                fill_price=filled_price,
                quantity=self.positions[listing_id],
                fee=fee,
            )

        return

    def on_market_update(self, timestamp: int, market_update: SchemaBase, market_recorder: MarketRecorder):
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

        market_recorder.log_market_event(
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
                new_intents = signal.process_new_tick(
                    data=self.listing_data, 
                    positions=self.signal_positions[signal], 
                    ticker_listing_id=listing_id,
                    timestamp=timestamp
                )
            else:
                new_intents = signal.process_new_tick(data=self.listing_data)
            
            if new_intents and len(new_intents) > 0:
                all_intents.extend(new_intents)
                # Log intents for debugging
                for intent in new_intents:
                    if isinstance(intent, BasketIntent):
                        for sub_intent, proportion in zip(intent.intents, intent.proportions):
                            logger.debug(
                                f"Intent generated: listing={sub_intent.listing.listing_id}, "
                                f"side={sub_intent.side}, confidence={sub_intent.confidence * proportion:.3f}, "
                                f"price={sub_intent.price}, flatten={getattr(sub_intent, 'flatten', False)}"
                            )
                    else:
                        logger.debug(
                            f"Intent generated: listing={intent.listing.listing_id}, "
                            f"side={intent.side}, confidence={intent.confidence:.3f}, "
                            f"price={intent.price}, flatten={getattr(intent, 'flatten', False)}"
                        )
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
                    # Order creation failed (bad midprice, no position to flatten, etc.)
                    # Clear pending flags so the signal isn't stuck
                    listing_id = intent.listing.listing_id
                    signals_for_listing = [s for s in self.signals
                                         if any(l.listing_id == listing_id for l in s.listings)]
                    for s in signals_for_listing:
                        if hasattr(s, '_entry_pending'):
                            s._entry_pending = False
                        if hasattr(s, '_exit_pending'):
                            s._exit_pending = False
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

        if midprice <= 0:
            return None

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
            size=order_size * FIXED_SIZE_SCALE,
            side=side,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC
        )

        return order


class LimitOrderOMS:
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
        position_scaling_factor: float = 0.5
    ):
        """Initialize LimitOrderOMS with position-aware sizing controls.
        
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
        """
        self.signals = signals
        # notional: user supplies in dollars, used as-is for order sizing calculations
        # max_position_notional: user supplies in dollars, stored in dollars (not scaled)
        self.notional = notional
        self.cash = starting_cash
        self.max_position_notional = max_position_notional  # Keep in dollars to match current_position_notional calculation
        self.position_aware_sizing = position_aware_sizing
        self.position_scaling_factor = position_scaling_factor

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

        # Track active orders: mapping of (listing_id, side) -> (client_oid, price)
        self.active_orders: dict[tuple[int, str], tuple[str, float]] = {}

        # Add order log to keep history of all submitted orders
        self.order_log: dict[str, Order] = {}
        self._order_counter: int = 0

        # Track elapsed ticks for each listing to control data appending frequency
        self.elapsed_ticks: dict[int, int] = {listing.listing_id: 0 for listing in all_listings}
    
    def _calculate_position_aware_order_size(
        self, 
        listing_id: int, 
        side: str, 
        base_notional: float, 
        confidence: float, 
        midprice: float
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
        # Base order size calculation
        # base_notional in dollars, midprice in dollars -> base_size in shares
        base_size = abs(float(base_notional * confidence / midprice))
        
        if not self.position_aware_sizing:
            return base_size
        
        # Get current position (in shares, not scaled)
        current_position = self.positions.get(listing_id, 0.0)
        # current_position in shares * midprice in dollars -> current_position_notional in dollars
        current_position_notional = abs(current_position * midprice)
        
        # Check max position notional limit
        if self.max_position_notional is not None:
            if current_position_notional >= self.max_position_notional:
                # At max position, don't add more in the same direction
                if (side == "B" and current_position > 0) or (side == "A" and current_position < 0):
                    return 0.0  # Don't add to position in same direction
        
        # Calculate position-aware scaling
        if self.max_position_notional is not None and self.max_position_notional > 0:
            # Scale down order size as position approaches max
            position_ratio = current_position_notional / self.max_position_notional
            # Reduce order size more aggressively as position grows
            # When position_ratio = 0, scaling = 1.0
            # When position_ratio = 1, scaling = position_scaling_factor
            scaling = 1.0 - (position_ratio * (1.0 - self.position_scaling_factor))
            scaling = max(scaling, self.position_scaling_factor)  # Don't go below minimum
        else:
            # No max position limit, but still scale based on absolute position
            # Use a soft limit based on base notional
            soft_limit = base_notional * 5.0  # 5x base notional as soft limit
            if soft_limit > 0:
                position_ratio = min(current_position_notional / soft_limit, 1.0)
                scaling = 1.0 - (position_ratio * (1.0 - self.position_scaling_factor))
                scaling = max(scaling, self.position_scaling_factor)
            else:
                scaling = 1.0
        
        # Additional scaling: reduce size if adding to position in same direction
        if (side == "B" and current_position > 0) or (side == "A" and current_position < 0):
            # Adding to existing position - reduce size more
            scaling *= self.position_scaling_factor
        
        # Increase size if reducing position (opposite direction)
        elif (side == "B" and current_position < 0) or (side == "A" and current_position > 0):
            # Reducing position - can be more aggressive
            scaling *= 1.5  # Boost size when reducing position
            scaling = min(scaling, 2.0)  # Cap at 2x
        
        return base_size * scaling

    def on_execution_report(self, timestamp: int, execution_report: OrderExecutionReport, recorder: MarketRecorder):
        """Handle execution reports from the exchange."""
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

        # Remove from active orders if filled or cancelled
        if execution_report.exec_type in [ExecType.TRADE, ExecType.CANCELED]:
            order_key = (listing_id, order.side)
            if order_key in self.active_orders and self.active_orders[order_key][0] == client_oid:
                del self.active_orders[order_key]

        if execution_report.exec_type in [ExecType.TRADE] and execution_report.filled_qty > 0:
            # Use order.side to determine position change direction
            # filled_qty is in scaled units (FIXED_SIZE_SCALE), convert to shares
            filled_qty = execution_report.filled_qty / FIXED_SIZE_SCALE
            filled_price = execution_report.filled_price / FIXED_PRICE_SCALE  # Scale the price
            position_change = filled_qty if order.side == "B" else -filled_qty

            # Update cash based on the trade
            trade_value = filled_qty * filled_price if filled_price > 0 else 0.0
            previous_cash = self.cash
            if order.side == "B":
                # Buying - cash decreases
                self.cash -= trade_value
                #print(f"Cash update - BUY: {previous_cash:.2f} -> {self.cash:.2f} (trade value: {trade_value:.2f})")
            else:
                # Selling - cash increases
                self.cash += trade_value
                #print(f"Cash update - SELL: {previous_cash:.2f} -> {self.cash:.2f} (trade value: {trade_value:.2f})")

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
                price=execution_report.mid_price / FIXED_PRICE_SCALE if execution_report.mid_price > 0 else filled_price,
                fill_price=filled_price,
                quantity=self.positions[listing_id],
                fee=execution_report.fee / (FIXED_PRICE_SCALE * FIXED_SIZE_SCALE),
            )

        return

    def on_market_update(self, timestamp: int, market_update: SchemaBase, market_recorder: MarketRecorder):
        """Process market update and generate/cancel orders as needed.

        Parameters
        ----------
        timestamp : int
            Event timestamp
        market_update : SchemaBase
            Market data update
        market_recorder : MarketRecorder
            Recorder for market and execution events
        """
        start_time = time.perf_counter()

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

        # Increment elapsed ticks for this listing
        self.elapsed_ticks[listing_id] += 1

        # Determine if we should append this data based on trade frequency
        min_trade_frequency = min(
            getattr(signal, 'trade_frequency', 1) for signal in self.signals
        )

        # Only append data if we've reached the trade frequency threshold
        should_append_data = (self.elapsed_ticks[listing_id] % min_trade_frequency == 0)

        # Initialize numpy arrays if needed
        if listing_id not in self.listing_data:
            self.listing_data[listing_id] = {}

        # Convert market data to dict and flatten levels
        market_dict = dataclasses.asdict(market_update)
        levels = market_dict.pop('levels', [])

        # Add flattened level data
        for i, level in enumerate(levels):
            market_dict[f'bidPrice{i}'] = level.get('bid_px', 0) / FIXED_PRICE_SCALE
            market_dict[f'askPrice{i}'] = level.get('ask_px', 0) / FIXED_PRICE_SCALE
            market_dict[f'bidSize{i}'] = level.get('bid_sz', 0) / FIXED_SIZE_SCALE
            market_dict[f'askSize{i}'] = level.get('ask_sz', 0) / FIXED_SIZE_SCALE
            market_dict[f'bidCount{i}'] = level.get('bid_ct', 0)
            market_dict[f'askCount{i}'] = level.get('ask_ct', 0)

        market_recorder.log_market_event(
            listing_id=listing_id,
            timestamp=timestamp,
            market_update=market_update,
            quantity=self.positions[listing_id],
        )

        # Add new data to numpy arrays (only if we should append data)
        if should_append_data:
            max_history_records = max([self.signals[i].max_lookback for i in range(len(self.signals))])

            for column, value in market_dict.items():
                if isinstance(value, str):
                    continue
                if value is None:
                    continue

                try:
                    float_value = float(value)
                except (ValueError, TypeError):
                    continue

                if column not in self.listing_data[listing_id]:
                    self.listing_data[listing_id][column] = np.array([float_value], dtype=np.float64)
                else:
                    current_array = self.listing_data[listing_id][column]
                    new_array = np.append(current_array, float_value)

                    if len(new_array) > max_history_records:
                        new_array = new_array[-max_history_records:]

                    self.listing_data[listing_id][column] = new_array

        # Generate intents from all signals
        all_intents = []
        
        for signal in self.signals:
            if isinstance(signal, PositionAwareSignal):
                new_intents = signal.process_new_tick(
                    data=self.listing_data, 
                    positions=self.signal_positions[signal], 
                    ticker_listing_id=listing_id,
                    timestamp=timestamp
                )
            else:
                new_intents = signal.process_new_tick(data=self.listing_data)
            
            if new_intents and len(new_intents) > 0:
                all_intents.extend(new_intents)
                # Log intents for debugging
                for intent in new_intents:
                    if isinstance(intent, BasketIntent):
                        for sub_intent, proportion in zip[tuple[Intent, float]](intent.intents, intent.proportions):
                            logger.debug(
                                f"Intent generated: listing={sub_intent.listing.listing_id}, "
                                f"side={sub_intent.side}, confidence={sub_intent.confidence * proportion:.3f}, "
                                f"price={sub_intent.price}, flatten={getattr(sub_intent, 'flatten', False)}"
                            )
                    else:
                        logger.debug(
                            f"Intent generated: listing={intent.listing.listing_id}, "
                            f"side={intent.side}, confidence={intent.confidence:.3f}, "
                            f"price={intent.price}, flatten={getattr(intent, 'flatten', False)}"
                        )

        # Only cancel/update orders when we have new intents
        # If no intents generated, keep existing orders active
        if len(all_intents) == 0:
            return []

        # Convert intents to orders and manage active orders
        orders = []
        cancel_orders = []

        # Track which orders we want to keep/update
        desired_orders: dict[tuple[int, str], tuple[float, float]] = {}  # (listing_id, side) -> (price, size)

        # Process intents to determine desired orders
        for intent in all_intents:

            if isinstance(intent, BasketIntent):
                # For basket intents, process each sub-intent
                for sub_intent, proportion in zip(intent.intents, intent.proportions):
                    if sub_intent.price is not None:
                        listing_id_intent = sub_intent.listing.listing_id
                        # Normalize side: "S" -> "A" to match order book convention
                        normalized_side = "A" if sub_intent.side in ["S", "A"] else "B"
                        order_key = (listing_id_intent, normalized_side)

                        # Calculate order size with position-aware adjustments
                        if listing_id_intent in self.listing_data:
                            bid_prices = self.listing_data[listing_id_intent]['bidPrice0']
                            ask_prices = self.listing_data[listing_id_intent]['askPrice0']
                            midprice = (bid_prices[-1] + ask_prices[-1]) / 2
                            order_size = self._calculate_position_aware_order_size(
                                listing_id=listing_id_intent,
                                side=normalized_side,
                                base_notional=self.notional * proportion,
                                confidence=sub_intent.confidence,
                                midprice=midprice
                            )
                        else:
                            continue

                        if order_size > 0:  # Only add if size > 0
                            desired_orders[order_key] = (sub_intent.price, order_size)
            else:
                if intent.price is not None:
                    listing_id_intent = intent.listing.listing_id
                    # Normalize side: "S" -> "A" to match order book convention
                    normalized_side = "A" if intent.side in ["S", "A"] else "B"
                    order_key = (listing_id_intent, normalized_side)

                    # Calculate order size with position-aware adjustments
                    if listing_id_intent in self.listing_data:
                        bid_prices = self.listing_data[listing_id_intent]['bidPrice0']
                        ask_prices = self.listing_data[listing_id_intent]['askPrice0']
                        midprice = (bid_prices[-1] + ask_prices[-1]) / 2
                        order_size = self._calculate_position_aware_order_size(
                            listing_id=listing_id_intent,
                            side=normalized_side,
                            base_notional=self.notional,
                            confidence=intent.confidence,
                            midprice=midprice
                        )
                    else:
                        continue

                    if order_size > 0:  # Only add if size > 0
                        desired_orders[order_key] = (intent.price, order_size)

        # Cancel orders that are no longer desired or have changed prices
        for order_key, (existing_oid, existing_price) in list(self.active_orders.items()):
            listing_id_key, side = order_key

            if order_key not in desired_orders:
                # Order no longer desired, cancel it
                listing = None
                for signal in self.signals:
                    for l in signal.listings:
                        if l.listing_id == listing_id_key:
                            listing = l
                            break
                    if listing:
                        break

                if listing:
                    cancel_orders.append(CancelOrder(
                        exchange_id=listing.exchange_id,
                        security_id=listing.security_id,
                        client_oid=existing_oid
                    ))
            else:
                desired_price, desired_size = desired_orders[order_key]
                # Check if price has changed significantly (more than 0.01% to avoid constant cancels)
                price_change_pct = abs(desired_price - existing_price) / existing_price if existing_price > 0 else float('inf')
                if price_change_pct > 0.0001:
                    # Price changed, cancel old order
                    listing = None
                    for signal in self.signals:
                        for l in signal.listings:
                            if l.listing_id == listing_id_key:
                                listing = l
                                break
                        if listing:
                            break

                    if listing:
                        cancel_orders.append(CancelOrder(
                            exchange_id=listing.exchange_id,
                            security_id=listing.security_id,
                            client_oid=existing_oid
                        ))

        # Create new orders for desired orders that don't exist or were cancelled
        for order_key, (desired_price, desired_size) in desired_orders.items():
            listing_id_key, side = order_key

            # Check if we already have this order active
            if order_key in self.active_orders:
                existing_oid, existing_price = self.active_orders[order_key]
                price_change_pct = abs(desired_price - existing_price) / existing_price if existing_price > 0 else float('inf')
                if price_change_pct <= 0.0001:
                    # Order already exists with same price, skip
                    continue

            # Find listing
            listing = None
            for signal in self.signals:
                for l in signal.listings:
                    if l.listing_id == listing_id_key:
                        listing = l
                        break
                if listing:
                    break

            if listing:
                # Create new limit order
                # side is already normalized to "B" or "A" from order_key
                self._order_counter += 1
                client_oid = f"limit_oms_{self._order_counter}_{listing_id_key}_{side}"
                order = Order(
                    exchange_id=listing.exchange_id,
                    security_id=listing.security_id,
                    client_oid=client_oid,
                    price=desired_price * FIXED_PRICE_SCALE,  # Convert to fixed point
                    size=desired_size * FIXED_SIZE_SCALE,  # Convert to fixed point
                    side=side,  # Already normalized to "B" or "A"
                    order_type=OrderType.LIMIT,
                    time_in_force=TimeInForce.GTC
                )

                self.order_log[client_oid] = order
                self.active_orders[order_key] = (client_oid, desired_price)
                orders.append(order)

        # Return both cancel orders and new orders
        return cancel_orders + orders