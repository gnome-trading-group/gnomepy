from typing import Callable
import pandas as pd
import numpy as np

class Action:
    def __init__(self, name: str, action_function: Callable):
        self.name = name
        self.action_function = action_function

    def apply(self, data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        if isinstance(data, pd.DataFrame):
            data[f'{self.name}_action'] = self.action_function(data)
        elif isinstance(data, np.ndarray):
            action_column = self.action_function(data)
            data = np.column_stack((data, action_column))
        return data

def single_ticker_rolling_mean_500_delta(data: pd.DataFrame, percentage_threshold: float = 0.005):
    cur_500_delta = (data['bidPrice0'].rolling(window=50000).mean() - data['bidPrice0'].rolling(window=50000).mean().shift(-1)).values
    last_500_delta = (data['bidPrice0'].rolling(window=50000).mean().shift(-1) - data['bidPrice0'].rolling(window=50000).mean().shift(-2)).values

    return np.where((last_500_delta > percentage_threshold) & (cur_500_delta <= percentage_threshold), -1, 
                    np.where((last_500_delta < -percentage_threshold) & (cur_500_delta >= -percentage_threshold), 1, 0))

def single_ticker_rolling_exp_mean_delta_alpha_0001(data: pd.DataFrame):
    last_ewm = data['bidPrice0'].ewm(alpha=.0001, min_periods=1000).mean().shift(-100)
    cur_ewm = data['bidPrice0'].ewm(alpha=.0001, min_periods=1000).mean()
    cur_exp_delta = (data['bidPrice0'].ewm(alpha=.0001, min_periods=1000).mean() - data['bidPrice0'].ewm(alpha=.0001, min_periods=1000).mean().shift(-1)).values
    last_exp_delta = (data['bidPrice0'].ewm(alpha=.0001, min_periods=1000).mean().shift(-1) - data['bidPrice0'].ewm(alpha=.0001, min_periods=1000).mean().shift(-2)).values

    return np.where((cur_exp_delta > 0) & (last_exp_delta < 0) & (cur_ewm < last_ewm), 1, np.where((cur_exp_delta < 0) & (last_exp_delta > 0) & (cur_ewm > last_ewm), -1, 0))

def single_ticker_rolling_exp_mean_delta_alpha_00001(data: pd.DataFrame):
    last_ewm = data['bidPrice0'].ewm(alpha=.00001, min_periods=1000).mean().shift(-100)
    cur_ewm = data['bidPrice0'].ewm(alpha=.00001, min_periods=1000).mean()
    cur_exp_delta = (data['bidPrice0'].ewm(alpha=.00001, min_periods=1000).mean() - data['bidPrice0'].ewm(alpha=.00001, min_periods=1000).mean().shift(-1)).values
    last_exp_delta = (data['bidPrice0'].ewm(alpha=.00001, min_periods=1000).mean().shift(-1) - data['bidPrice0'].ewm(alpha=.00001, min_periods=1000).mean().shift(-2)).values

    return np.where((cur_exp_delta > 0) & (last_exp_delta < 0) & (cur_ewm < last_ewm), 1, np.where((cur_exp_delta < 0) & (last_exp_delta > 0) & (cur_ewm > last_ewm), -1, 0))

def single_ticker_moving_average_crossover(data: pd.DataFrame, short_window: int = 50, long_window: int = 200, percentage_threshold: float = 0.005):
    short_mavg = data['bidPrice0'].rolling(window=short_window).mean()
    long_mavg = data['bidPrice0'].rolling(window=long_window).mean()
    mavg_diff = short_mavg - long_mavg

    return np.where(mavg_diff > percentage_threshold, 1, 
                    np.where(mavg_diff < -percentage_threshold * 2, -1,  # Reduce sell frequency by requiring a larger negative threshold
                             np.where((short_mavg.shift(1) <= long_mavg.shift(1)) & (short_mavg > long_mavg), 1, 
                                      np.where((short_mavg.shift(1) >= long_mavg.shift(1)) & (short_mavg < long_mavg), -1, 0))))

global_actions = {
    "single_ticker_rolling_mean_500_delta": Action(
        name="single_ticker_rolling_mean_500_delta",
        action_function=single_ticker_rolling_mean_500_delta
    ),
    "single_ticker_rolling_exp_mean_delta_alpha_00001": Action(
        name="single_ticker_rolling_exp_mean_delta_alpha_00001",
        action_function=single_ticker_rolling_exp_mean_delta_alpha_00001
    ),
    "single_ticker_rolling_exp_mean_delta_alpha_0001": Action(
        name="single_ticker_rolling_exp_mean_delta_alpha_0001",
        action_function=single_ticker_rolling_exp_mean_delta_alpha_0001
    ),
    "single_ticker_moving_average_crossover": Action(
        name="single_ticker_moving_average_crossover",
        action_function=single_ticker_moving_average_crossover
    )
}
