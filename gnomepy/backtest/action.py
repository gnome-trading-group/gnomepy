from typing import Callable
import pandas as pd
import numpy as np

class Action:
    def __init__(self, name: str, required_signals: list[dict], action_function: Callable):
        self.name = name
        self.required_signals = required_signals
        self.action_function = action_function

    def apply(self, data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        if isinstance(data, pd.DataFrame):
            data[f'{self.name}_action'] = self.action_function(data)
        elif isinstance(data, np.ndarray):
            action_column = self.action_function(data)
            data = np.column_stack((data, action_column))
        return data

def bidPrice0_rolling_mean_10_under_100(data: pd.DataFrame):
    return np.where(data['bidPrice0_rolling_mean_10'].values < data['bidPrice0_rolling_mean_100'].values, 1, 
                    np.where(data['bidPrice0_rolling_mean_10'].values > data['bidPrice0_rolling_mean_100'].values, -1, 0))

def askPrice0_rolling_std_20_above_threshold(data: pd.DataFrame, threshold: float = 0.35):
    return np.where(data['askPrice0_rolling_std_20'].values > threshold, 1, 
                    np.where(data['askPrice0_rolling_std_20'].values < threshold, -1, 0))

def bid_ask_spread_narrowing(data: pd.DataFrame, spread_threshold: float = 0.01):
    spread = data['askPrice0'].values - data['bidPrice0'].values
    return np.where(spread < spread_threshold, 1, np.where(spread > spread_threshold, -1, 0))

def bidPrice0_rolling_mean_50_over_500(data: pd.DataFrame, percentage_threshold: float = 0.001):
    difference = (data['bidPrice0_rolling_mean_50'].values - data['bidPrice0_rolling_mean_500'].values) / data['bidPrice0_rolling_mean_500'].values
    return np.where(difference > percentage_threshold, -1, 
                    np.where(difference < -percentage_threshold, 1, 0))

global_actions = {
    "bidPrice0_rolling_mean_10_under_100": Action(
        name="bidPrice0_rolling_mean_10_under_100",
        required_signals=[
            {"signal_name": "rolling_mean_10", "columns": ["bidPrice0"]},
            {"signal_name": "rolling_mean_100", "columns": ["bidPrice0"]}
        ],
        action_function=bidPrice0_rolling_mean_10_under_100
    ),
    "askPrice0_rolling_std_20_above_threshold": Action(
        name="askPrice0_rolling_std_20_above_threshold",
        required_signals=[
            {"signal_name": "rolling_std_20", "columns": ["askPrice0"]}
        ],
        action_function=askPrice0_rolling_std_20_above_threshold
    ),
    "bid_ask_spread_narrowing": Action(
        name="bid_ask_spread_narrowing",
        required_signals=[],
        action_function=bid_ask_spread_narrowing
    ),
    "bidPrice0_rolling_mean_50_over_500": Action(
        name="bidPrice0_rolling_mean_50_over_500",
        required_signals=[
            {"signal_name": "rolling_mean_50", "columns": ["bidPrice0"]},
            {"signal_name": "rolling_mean_500", "columns": ["bidPrice0"]}
        ],
        action_function=bidPrice0_rolling_mean_50_over_500
    )
}
