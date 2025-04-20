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

        # If the action is to buy (action value is 1), multiply by askPrice0
        # If the action is to sell (action value is -1), multiply by bidPrice0
        data[f"{self.action.name}_cash_action"] = np.where(
            data[f"{self.action.name}_action"] == 1,
            data[f"{self.action.name}_action"] * -data[f"askPrice0"],
            np.where(
                data[f"{self.action.name}_action"] == -1,
                data[f"{self.action.name}_action"] * -data[f"bidPrice0"],
                0  # If action is neither buy nor sell, cash is 0
            )
        )

        ############################## Trying to vectorize PnL
        # Initialize cash_balance and equity_position columns
        # data[f"{self.action.name}_cash_balance"] = self.starting_cash
        # data[f"{self.action.name}_equity_position"] = 0

        # # Calculate cash_balance and equity_position for each row
        # data[f"{self.action.name}_cash_balance"] = self.starting_cash + data[f"{self.action.name}_cash_action"].cumsum()
        # data[f"{self.action.name}_equity_position"] = data[f"{self.action.name}_position"].cumsum()

        # # Enforce the constraint that cash_balance > 0 to be able to buy
        # negative_cash_balance_indices = (data[f"{self.action.name}_cash_balance"] < 0) | ((data[f"{self.action.name}_cash_balance"] +  data[f"{self.action.name}_cash_action"]) < 0)
        # negative_equity_position_indices = (data[f"{self.action.name}_equity_position"] < 0) | ((data[f"{self.action.name}_equity_position"] + data[f"{self.action.name}_position"]) < 0)

        # # Revert actions where cash balance is negative
        # data.loc[negative_cash_balance_indices, f"{self.action.name}_cash_action"] = 0
        # data.loc[negative_cash_balance_indices, f"{self.action.name}_position"] = 0

        # # Revert actions where equity position is negative
        # data.loc[negative_equity_position_indices, f"{self.action.name}_cash_action"] = 0
        # data.loc[negative_equity_position_indices, f"{self.action.name}_position"] = 0
        
        # data.loc[negative_cash_balance_indices, f"{self.action.name}_equity_position"] = data[f"{self.action.name}_equity_position"].shift(1)
        # data.loc[negative_cash_balance_indices, f"{self.action.name}_cash_balance"] = data[f"{self.action.name}_cash_balance"].shift(1)
        # data.loc[negative_equity_position_indices, f"{self.action.name}_equity_position"] = data[f"{self.action.name}_equity_position"].shift(1)
        # data.loc[negative_equity_position_indices, f"{self.action.name}_cash_balance"] = data[f"{self.action.name}_cash_balance"].shift(1)

        #         # Calculate cash_balance and equity_position for each row
        # data[f"{self.action.name}_cash_balance"] = self.starting_cash + data[f"{self.action.name}_cash_action"].cumsum()
        # data[f"{self.action.name}_equity_position"] = data[f"{self.action.name}_position"].cumsum()

        ############################## Trying to vectorize PnL

        ############################# Implement guardrails on cash and equity
        # Iterate through each row in data and calculate cash_balance and equity_position
        # cash_balance = self.starting_cash
        # equity_position = 0

        # for index, row in data.iterrows():
        #     cash_action = row[f"{self.action.name}_cash_action"]
        #     position = row[f"{self.action.name}_position"]

        #     # Calculate new cash balance and equity position
        #     new_cash_balance = cash_balance + cash_action
        #     new_equity_position = equity_position + position

        #     # Ensure cash balance and equity position are not negative
        #     if new_cash_balance < 0:
        #         new_cash_balance = cash_balance
        #         new_equity_position = equity_position
        #         data.at[index, f"{self.action.name}_cash_action"] = 0
        #         data.at[index, f"{self.action.name}_position"] = 0
        #     elif new_equity_position < 0:
        #         new_cash_balance = cash_balance
        #         new_equity_position = equity_position
        #         data.at[index, f"{self.action.name}_cash_action"] = 0
        #         data.at[index, f"{self.action.name}_position"] = 0
        #     else:
        #         cash_balance = new_cash_balance
        #         equity_position = new_equity_position

        #     # Update the data with the new values
        #     data.at[index, f"{self.action.name}_cash_balance"] = cash_balance
        #     data.at[index, f"{self.action.name}_equity_position"] = equity_position
        ############################# Implement guardrails on cash and equity

        # # Calculate PnL
        # data[f"{self.action.name}_cash_balance"] = self.starting_cash
        # data[f"{self.action.name}_equity_position"] = 0

        # Calculate cash_balance and equity_position for each row
        data[f"{self.action.name}_cash_balance"] = self.starting_cash + data[f"{self.action.name}_cash_action"].cumsum()
        data[f"{self.action.name}_equity_position"] = data[f"{self.action.name}_position"].cumsum()


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
