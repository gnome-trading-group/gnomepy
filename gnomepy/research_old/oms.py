import pandas as pd
import numpy as np 
from gnomepy.data.types import *
from gnomepy.research_old.strategy import *
from gnomepy.research_old.trade_signal import TradeSignal, BasketTradeSignal


# Order management class
# TODO: Mason you can help fill this out, I am only going to write basic functionality
class OMS:

    def __init__(self, strategies: list[Strategy], notional: float, starting_cash: float):
        self.strategies = strategies
        self.notional = notional
        self.cash = starting_cash
        self.positions = {listing: 0.0 for strategy in self.strategies 
                                   for listing in strategy.listings}
        self.open_orders = []
        self.order_log = []  # Internal order log to track filled orders as {strategy_hash: order} dicts

        # Track positions per strategy
        self.strategy_positions = {}
        for strategy in strategies:
            self.strategy_positions[str(strategy)] = {
                'position_type': None,  # 'positive_mean_reversion', 'negative_mean_reversion', None
                'beta_vector': None,    # The beta vector used for this position
                'entry_zscore': None,   # The z-score when we entered
                'timestamp': None       # When we entered
            }
        
    def process_signals(self, signals: list[TradeSignal | BasketTradeSignal], lisings_lob_data: dict[Listing, pd.DataFrame]):
        """Process incoming signals and generate filled orders"""
        filled_order_log = []

        for signal in signals:
            if isinstance(signal, BasketTradeSignal):
                filled_orders = self._execute_basket_signal(signal, lisings_lob_data)
                if filled_orders:
                    for order in filled_orders:
                        filled_order_log.append({str(signal.strategy): order})
                        self.order_log.append({str(signal.strategy): order})
                    self._update_strategy_position(signal)

            else:
                filled_order = self._execute_single_signal(signal, lisings_lob_data)
                if filled_order:
                    # For single signals, we need to find the strategy that generated it
                    # This is a simplified approach - in practice you might want to store the strategy with each signal
                    strategy_hash = "single_signal"  # Placeholder
                    filled_order_log.append({strategy_hash: filled_order})
                    self.order_log.append({strategy_hash: filled_order})
                    # Note: _update_strategy_position is only called for BasketTradeSignals

        return filled_order_log

    def _execute_single_signal(self, signal: TradeSignal, listings_lob_data: dict[Listing, pd.DataFrame]):
        """Execute a single signal"""
        scaled_notional = signal.confidence * self.notional
        
        order = Order(
            listing=signal.listing,
            size=None,
            status=Status.OPEN,
            action=signal.action,
            price=None,
            cash_size=scaled_notional,
            type=OrderType.MARKET,
            timestampOpened=listings_lob_data[signal.listing].iloc[-1]['timestampEvent'],
            signal=signal
        )

        return self.simulate_lob(order=order, lob_data=listings_lob_data[signal.listing])

    def _execute_basket_signal(self, signal: BasketTradeSignal, listings_lob_data: dict[Listing, pd.DataFrame]):
        """Execute a basket of signals"""
        if not signal.strategy.validate_signal(signal, self.strategy_positions[str(signal.strategy)]):
            return None

        filled_orders = []
        
        # Validate we have data for all listings
        for subsignal in signal.signals:
            if subsignal.listing not in listings_lob_data:
                return None
            if len(listings_lob_data[subsignal.listing]) == 0:
                return None
                
        # Try to fill all orders
        for i, subsignal in enumerate(signal.signals):
            scaled_notional = subsignal.confidence * self.notional
            
            order = Order(
                listing=subsignal.listing,
                size=None,
                status=Status.OPEN,
                action=subsignal.action,
                price=None,
                cash_size=scaled_notional * abs(signal.proportions.flatten()[i]),
                type=OrderType.MARKET,
                timestampOpened=listings_lob_data[subsignal.listing].iloc[-1]['timestampEvent'],
                signal=signal
            )

            filled_order = self.simulate_lob(order=order, lob_data=listings_lob_data[subsignal.listing])
            if filled_order:
                filled_orders.append(filled_order)
            else:
                return None  # Return None if any order fails to fill

        # Only return filled orders if all orders were filled
        if len(filled_orders) == len(signal.signals):
            return filled_orders
        else:
            return None

    def _update_strategy_position(self, signal: BasketTradeSignal):
        """Update the strategy position state after executing a signal"""
        strategy = signal.strategy
        self.strategy_positions[str(strategy)].update({
            'position_type': signal.signal_type,
            'beta_vector': signal.proportions,
            'entry_zscore': None,  # TODO: Add zscore tracking
            'timestamp': None  # TODO: Add timestamp tracking
        })

    def process_open_orders(self, listings_lob_data: dict[Listing, pd.DataFrame]):
        """Process any open orders that haven't been fully filled yet"""
        filled_order_log = []
        remaining_open_orders = []

        for order in self.open_orders:
            filled_order = self.simulate_lob(order=order, lob_data=listings_lob_data[order.listing])
            if filled_order is not None:
                # Add order to logs with strategy hash
                filled_order_log.append({str(order.signal.strategy): filled_order})
                self.order_log.append({str(order.signal.strategy): filled_order})
            else:
                remaining_open_orders.append(order)

        self.open_orders = remaining_open_orders
        return filled_order_log

    def simulate_lob(self, order: Order, lob_data: pd.DataFrame):
        """
        Simulate order execution with realistic market impact and slippage modeling.
        
        The model incorporates several components of execution costs:
        1. Temporary price impact: Square root model based on order size vs. available liquidity
           - Follows literature that suggests impact scales with square root of order size
           - Impact factor (0.1) should be calibrated to historical data
           - Impact is scaled by market volatility
        
        2. Competition for liquidity:
           - Base competition reduces available size by 10-70%
           - Additional penalty for large orders relative to level size
           - Reflects that larger orders are harder to execute efficiently
        
        3. Multi-level impact:
           - Walks the order book to simulate realistic fills
           - Each level has its own impact calculation
           - Prevents unrealistic assumptions about deep liquidity
        """
        
        # Calculate volatility adjustment factor (increases impact in volatile periods)
        # Using simple rolling std of mid prices as volatility estimate
        mid_prices = (lob_data['askPrice0'] + lob_data['bidPrice0'])/2
        vol = mid_prices.rolling(20).std().iloc[-1] / mid_prices.iloc[-1]  # Normalized volatility
        vol_factor = 1 + vol  # Scale impact up in volatile periods
            
        if order.size != None:
            remaining_size = order.size
        else:
            if order.type == OrderType.MARKET and order.action == Action.BUY:
                remaining_size = order.cash_size / lob_data['askPrice0'].iloc[-1]
            elif order.type == OrderType.MARKET and order.action == Action.SELL:
                remaining_size = order.cash_size / lob_data['bidPrice0'].iloc[-1]
            ## TODO: Implement other scenarios

        filled_size = 0
        weighted_price = 0

        # Get latest LOB snapshot for order simulation
        latest_lob = lob_data.iloc[-1]
        
        # Debug print for the first trade
        if not hasattr(self, '_printed_lob_debug'):
            print(f"\n=== OMS simulate_lob debug (first trade) ===")
            print(f"Order: {order.action.value} {order.listing}")
            print(f"LOB data shape: {lob_data.shape}")
            print(f"Latest LOB row:")
            print(latest_lob[['timestampEvent', 'bidPrice0', 'askPrice0', 'bidSize0', 'askSize0']])
            print(f"Available price columns: {[col for col in lob_data.columns if 'Price' in col][:10]}")
            self._printed_lob_debug = True

        # Look through order book levels until we fill the full size
        for level in range(10):  # Assuming 10 levels in the order book
            if order.action == Action.BUY:
                price = latest_lob[f'askPrice{level}'] if order.type == OrderType.MARKET else order.price
                available_size = latest_lob[f'askSize{level}']
                
                # Temporary price impact using square root model
                # Impact increases with order size and decreases with market liquidity
                # Formula: impact = λ * σ * sqrt(V_order / V_market) where:
                # λ is the impact factor
                # σ is the volatility scaling factor
                impact_factor = 0.1 * vol_factor  # Base impact scaled by volatility
                temp_impact = impact_factor * np.sqrt(remaining_size / available_size) if available_size > 0 else 0
                price = price * (1 + temp_impact)

            elif order.action == Action.SELL:
                price = latest_lob[f'bidPrice{level}'] if order.type == OrderType.MARKET else order.price
                available_size = latest_lob[f'bidSize{level}']
                
                # Similar impact model for sells, but negative impact
                impact_factor = 0.1 * vol_factor
                temp_impact = impact_factor * np.sqrt(remaining_size / available_size) if available_size > 0 else 0
                price = price * (1 - temp_impact)

            # Skip empty levels
            if available_size <= 0:
                continue
                
            # Model competition for liquidity
            # Base competition factor reduces available size by 10-70%
            # Additional size penalty for large orders
            # Formula: final_factor = max(0.3, base_competition - size_penalty)
            base_competition = np.random.uniform(0.3, 0.9)
            size_penalty = 0.1 * (remaining_size / available_size) if available_size > 0 else 0
            competition_factor = max(0.3, base_competition - size_penalty)
            available_size = available_size * competition_factor
                
            # Calculate fill at this level
            fill_size = min(remaining_size, available_size)
            filled_size += fill_size
            weighted_price += price * fill_size
            remaining_size -= fill_size
            
            if remaining_size <= 0:
                break

        # Return unfilled if we couldn't fill any size
        if filled_size == 0:
            return None
        
        # Calculate total cash with correct sign based on action and add fees
        total_cash = weighted_price
        fee = total_cash * 4.5e-4  # Calculate fee as 0.045% of trade value
        
        if order.action == Action.BUY:
            total_cash = -(total_cash + fee)  # Negative for buys, add fee
            self.positions[order.listing] += filled_size
        else:  # sell
            total_cash = total_cash - fee  # Positive for sells, subtract fee
            self.positions[order.listing] -= filled_size
            
        self.cash += total_cash

        if remaining_size <= 0:
            # If fully filled, update the original order
            order.size = filled_size
            order.price = weighted_price/filled_size
            order.cash_size = total_cash
            order.close(latest_lob['timestampEvent'])
            return order
        else:
            # If partially filled, create new order for filled portion
            filled_order = Order(
                listing=order.listing,
                action=order.action,
                type=order.type,
                size=filled_size,
                price=weighted_price/filled_size,
                cash_size=total_cash,
                status=Status.FILLED,
                timestampOpened=order.timestampOpened,
                signal=order.signal
            )
            filled_order.close(latest_lob['timestampEvent'])

            # Create remaining order for unfilled portion
            remaining_order = Order(
                listing=order.listing,
                action=order.action, 
                type=order.type,
                size=remaining_size,
                price=order.price,
                cash_size=None,
                status=Status.OPEN,
                timestampOpened=order.timestampOpened
            )
            self.open_orders.append(remaining_order)

            return filled_order