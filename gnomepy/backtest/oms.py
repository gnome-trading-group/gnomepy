import pandas as pd
import numpy as np 
from gnomepy.data.types import *
from gnomepy.backtest.strategy import *


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
        
    def process_signals(self, signals: list[Signal | BasketSignal], lisings_lob_data: dict[Listing, pd.DataFrame]):
        """Process incoming signals and generate filled orders"""
        filled_order_log = []

        for signal in signals:
            if isinstance(signal, BasketSignal):
                filled_orders = self._execute_basket_signal(signal, lisings_lob_data)
                if filled_orders:
                    for order in filled_orders:
                        filled_order_log.append({str(signal.strategy): order})
                        self.order_log.append({str(signal.strategy): order})
                    self._update_strategy_position(signal)

            else:
                filled_order = self._execute_single_signal(signal, lisings_lob_data)
                if filled_order:
                    filled_order_log.append({str(signal.strategy): filled_order})
                    self.order_log.append({str(signal.strategy): filled_order})
                    self._update_strategy_position(signal)

        return filled_order_log

    def _execute_single_signal(self, signal: Signal, listings_lob_data: dict[Listing, pd.DataFrame]):
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
            timestampOpened=listings_lob_data[signal.listing].iloc[-1]['timestampEvent']
        )

        return self.simulate_lob(order=order, lob_data=listings_lob_data[signal.listing].iloc[-1])

    def _execute_basket_signal(self, signal: BasketSignal, listings_lob_data: dict[Listing, pd.DataFrame]):
        """Execute a basket of signals"""
        if not signal.strategy.validate_signal(signal, self.strategy_positions[str(signal.strategy)]):
            print(f"Signal validation failed for {signal.signal_type}")
            print(f"Signal attempted to {signal.signal_type}, when the strategy position was {self.strategy_positions[str(signal.strategy)]}")
            return None

        filled_orders = []
        for i, subsignal in enumerate(signal.signals):
            scaled_notional = subsignal.confidence * self.notional
            
            order = Order(
                listing=subsignal.listing,
                size=None,
                status=Status.OPEN,
                action=subsignal.action,
                price=None,
                cash_size=scaled_notional * abs(signal.proportions[i][0]),
                type=OrderType.MARKET,
                timestampOpened=listings_lob_data[subsignal.listing].iloc[-1]['timestampEvent']
            )

            filled_order = self.simulate_lob(order=order, lob_data=listings_lob_data[subsignal.listing].iloc[-1])
            if filled_order:
                filled_orders.append(filled_order)

        return filled_orders

    def _update_strategy_position(self, signal: BasketSignal):
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
            filled_order = self.simulate_lob(order=order, lob_data=listings_lob_data[order.listing].iloc[-1])
            if filled_order is not None:
                filled_order_log.append(filled_order)
                self.order_log.append(filled_order)  # Add to internal log
            else:
                remaining_open_orders.append(order)

        self.open_orders = remaining_open_orders
        return filled_order_log

    def simulate_lob(self, order: Order, lob_data: pd.DataFrame):
        
        if order.size != None:
            remaining_size = order.size
            print(f"Order has explicit size: {remaining_size}")

        else:
            if order.type == OrderType.MARKET and order.action == Action.BUY:
                remaining_size = order.cash_size / lob_data['askPrice0'].item()

            elif order.type == OrderType.MARKET and order.action == Action.SELL:
                remaining_size = order.cash_size / lob_data['bidPrice0'].item()

            ## TODO: Implement other scenarios

        filled_size = 0
        weighted_price = 0

        print(f"\nAttempting to fill order of size {remaining_size}")
        # Look through order book levels until we fill the full size
        for level in range(10):  # Assuming 10 levels in the order book
            if order.action == Action.BUY:
                price = lob_data[f'askPrice{level}'].item() if order.type == OrderType.MARKET else order.price
                available_size = lob_data[f'askSize{level}'].item()
                print(f"Level {level} ASK: {available_size} @ {price}")

            elif order.action == Action.SELL:
                price = lob_data[f'bidPrice{level}'].item() if order.type == OrderType.MARKET else order.price
                available_size = lob_data[f'bidSize{level}'].item()
                print(f"Level {level} BID: {available_size} @ {price}")

            # Skip if no size available at this level
            if available_size <= 0:
                print(f"No size available at level {level}, skipping")
                continue
                
            # Randomly reduce available size to simulate competition
            # We can get between 30% to 90% of the displayed size
            competition_factor = np.random.uniform(0.9, 1.0)
            available_size = available_size * competition_factor
            print(f"After competition factor {competition_factor:.2f}, available size: {available_size:.2f}")
                
            # Calculate how much we can fill at this level
            fill_size = min(remaining_size, available_size)
            filled_size += fill_size
            weighted_price += price * fill_size
            remaining_size -= fill_size
            print(f"Filled {fill_size:.2f} @ {price}, remaining: {remaining_size:.2f}")
            
            # Break if we've filled the entire order
            if remaining_size <= 0:
                print("Order fully filled")
                break

        # Return unfilled if we couldn't fill any size
        if filled_size == 0:
            print("Could not fill any size, returning None")
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

        print(f"\nFinal fill: {filled_size:.2f} @ {weighted_price/filled_size:.2f}")
        print(f"Total cash flow: {total_cash:.2f} (including {fee:.2f} fees)")

        if remaining_size <= 0:
            # If fully filled, update the original order
            order.size = filled_size
            order.price = weighted_price/filled_size
            order.cash_size = total_cash
            order.close(lob_data['timestampEvent'])
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
                timestampOpened=order.timestampOpened
            )
            filled_order.close(lob_data['timestampEvent'])

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
    
    def compute_portfolio_metrics(self, price_data: dict[Listing, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute portfolio metrics using order history and price data.
        Calculates metrics like P&L, profit factor, drawdown, win rate etc.
        
        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            DataFrame containing portfolio metrics and trade history DataFrame
        """
        # Create history dataframe from order log
        history = pd.DataFrame([{
            'timestamp': order.timestampClosed,
            'listing': order.listing.security_id,
            'action': order.action.value,
            'size': order.size,
            'price': order.price,
            'cash_flow': order.cash_size,
            'strategy': strategy_hash
        } for order_dict in self.order_log 
          for strategy_hash, order in order_dict.items()])
        
        if len(history) == 0:
            return pd.DataFrame(), pd.DataFrame()
            
        # Sort by timestamp
        history = history.sort_values('timestamp')
        
        # Calculate P&L per trade
        history['pl'] = history['cash_flow']
        history['cum_pl'] = history.groupby('strategy')['pl'].cumsum()
        
        # Calculate metrics per strategy
        metrics_list = []
        for strategy_name in history['strategy'].unique():
            strategy_history = history[history['strategy'] == strategy_name]
            
            total_trades = len(strategy_history)
            winning_trades = len(strategy_history[strategy_history['pl'] > 0])
            losing_trades = len(strategy_history[strategy_history['pl'] < 0])
            
            metrics = {
                'strategy': strategy_name,
                'total_pl': strategy_history['pl'].sum(),
                'profit_factor': abs(strategy_history[strategy_history['pl'] > 0]['pl'].sum()) / abs(strategy_history[strategy_history['pl'] < 0]['pl'].sum()) if abs(strategy_history[strategy_history['pl'] < 0]['pl'].sum()) != 0 else float('inf'),
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'avg_pl_per_trade': strategy_history['pl'].mean(),
                'std_pl_per_trade': strategy_history['pl'].std(),
                'max_drawdown': (strategy_history['cum_pl'] - strategy_history['cum_pl'].expanding().max()).min(),
                'sharpe_ratio': strategy_history['pl'].mean() / strategy_history['pl'].std() * np.sqrt(252) if strategy_history['pl'].std() != 0 else 0,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades
            }
            metrics_list.append(metrics)
        
        return pd.DataFrame(metrics_list), history