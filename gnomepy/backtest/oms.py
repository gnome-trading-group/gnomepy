import pandas as pd
import numpy as np 
from gnomepy.data.types import *
from gnomepy.backtest.strategy import *


# Order management class
# TODO: Mason you can help fill this out, I am only going to write basic functionality
class OMS:


    def process_signals(self, signals: list[Signal | BasketSignal], lisings_lob_data: dict[Listing, pd.DataFrame]):

        # TODO: Determine how to figure this out idk
        notional = 100

        filled_order_log = []

        # Process each signal one by one 
        for signal in signals:

            # Unique BasketSignal case
            if type(signal) is BasketSignal:
                
                # Iterate through each signal in Basket
                for i in range(0, len(signal.signals)):
                    scaled_notional = signal.signals[i].confidence * notional

                    # Create order
                    order = Order(listing=signal.signals[i].listing, size=None,
                                  status=Status.OPEN, action=signal.signals[i].action,
                                  price=None, cash_size=scaled_notional*abs(signal.proportions[i]))
                    
                    # Process Order
                    filled_order = self.simulate_lob(order=order, lob_data=lisings_lob_data[signal.signals[i].listing].iloc[-1])
                    if filled_order is not None:
                        filled_order_log.append(filled_order)
            else:
                pass



    def simulate_lob(self, order: Order, lob_data: pd.DataFrame):
        
        if order.size != None:
            remaining_size = order.size

        else:
            if order.type == OrderType.MARKET and order.action == Action.BUY:
                remaining_size = lob_data['askPrice0'].item() / order.cash_size
            elif order.type == OrderType.MARKET and order.action == Action.SELL:
                remaining_size = lob_data['bidPrice0'].item() / order.cash_size

            ## TODO: Implement other scenarios

        filled_size = 0
        weighted_price = 0

        # Look through order book levels until we fill the full size
        for level in range(10):  # Assuming 10 levels in the order book
            if order.action == Action.BUY:
                price = lob_data[f'askPrice{level}'] if order.type == OrderType.MARKET else order.price
                available_size = lob_data[f'askSize{level}']

            elif order.action == Action.SELL:
                price = lob_data[f'bidPrice{level}'] if order.type == OrderType.MARKET else order.price
                available_size = lob_data[f'bidSize{level}']


            # Skip if no size available at this level
            if available_size.item() <= 0:
                continue
                
            # Randomly reduce available size to simulate competition
            # We can get between 30% to 90% of the displayed size
            competition_factor = np.random.uniform(0.6, 1.0)
            available_size = available_size.item() * competition_factor
                
            # Calculate how much we can fill at this level
            fill_size = min(remaining_size, available_size)
            filled_size += fill_size
            weighted_price += price.item() * fill_size
            remaining_size -= fill_size

            # Break if we've filled the entire order
            if remaining_size <= 0:
                break

        ## TODO: AFTER THIS WE NEED TO RETURN FILLED ORDER 
        ## AND PROBABLY DO OTHER THINGS
        ## SORRY RAN OUT OF TIME