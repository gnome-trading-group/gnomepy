from gnomepy.data.types import *
from gnomepy.backtest.archive.signal import *
from gnomepy.backtest.action import *
import pandas as pd
import numpy as np

class Strategy:
    def __init__(self, name: str, action: Action, minimum_ticker_cycle: int, starting_cash: float):
        self.name = name
        self.action = action
        self.minimum_ticker_cycle = minimum_ticker_cycle
        self.submit_order_buffer = 3
        self.fill_order_buffer = 3
        self.starting_cash = starting_cash
    
    def assess_trades(self, data: pd.DataFrame | np.ndarray):

        # Translate the action column into realized trades
        data[f"{self.action.name}_position"] = data[f"{self.action.name}_action"]

        # Implement submit_order_buffer: delay the action by 3 ticks
        data[f"{self.action.name}_realized_action"] = data[f"{self.action.name}_action"].shift(self.submit_order_buffer).fillna(0)

        # If the realized action is to buy (action value is 1), multiply by askPrice0
        # If the realized action is to sell (action value is -1), multiply by bidPrice0
        data[f"{self.action.name}_cash_action"] = np.where(
            data[f"{self.action.name}_realized_action"] == 1,
            data[f"{self.action.name}_realized_action"] * -data[f"askPrice0"],
            np.where(
                data[f"{self.action.name}_realized_action"] == -1,
                data[f"{self.action.name}_realized_action"] * -data[f"bidPrice0"],
                0  # If action is neither buy nor sell, cash is 0
            )
        )

        # Calculate cash_balance and equity_position for each row
        data[f"{self.action.name}_cash_balance"] = self.starting_cash + data[f"{self.action.name}_cash_action"].cumsum()
        data[f"{self.action.name}_equity_position"] = data[f"{self.action.name}_realized_action"].cumsum()

        # Liquidate all equity on last step 
        total_positions = data.loc[data.index[-2], f"{self.action.name}_equity_position"]

        # If the sum is positive, sell all positions at bidPrice0
        if total_positions > 0:
            data.loc[data.index[-1], f"{self.action.name}_equity_position"] = -total_positions
            data.loc[data.index[-1], f"{self.action.name}_cash_balance"] = data.loc[data.index[-1], f"{self.action.name}_cash_balance"] + total_positions * data.loc[data.index[-1], f"bidPrice0"]

        # If the sum is negative, buy out the position at askPrice0, then sell at bidPrice0
        elif total_positions < 0:
            data.loc[data.index[-1], f"{self.action.name}_equity_position"] = total_positions
            data.loc[data.index[-1], f"{self.action.name}_cash_balance"] = data.loc[data.index[-1], f"{self.action.name}_cash_balance"] + total_positions * data.loc[data.index[-1], f"bidPrice0"]

        return data, data[f"{self.action.name}_cash_balance"].sum(), data[f"{self.action.name}_equity_position"].sum()

    def execute(self, data: pd.DataFrame):

        # First add cash to data
        data['cash'] = self.starting_cash

        #  Apply the actual action function for the chosen strategy
        data = self.action.apply(data)

        # Now we need to assess the trades that were supposed to be executed
        data, cash, position = self.assess_trades(data)

        # Adjust cash with starting cash
        total_cash = self.starting_cash + cash

        return (data, total_cash, position)
