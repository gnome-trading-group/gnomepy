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

def single_ticker_rolling_mean_500_delta(data: pd.DataFrame):
    last_500_mean = data['bidPrice0'].rolling(window=5000).mean().shift(-1)
    cur_500_mean = data['bidPrice0'].rolling(window=5000).mean()
    cur_500_delta = (cur_500_mean - cur_500_mean.shift(-1)).values
    last_500_delta = (cur_500_mean.shift(-1) - cur_500_mean.shift(-2)).values

    return np.where((cur_500_delta > 0) & (last_500_delta < 0) & (cur_500_mean < last_500_mean), 1, 
                    np.where((cur_500_delta < 0) & (last_500_delta > 0) & (cur_500_mean > last_500_mean), -1, 0))

def single_ticker_rolling_exp_mean_delta_alpha_0001(data: pd.DataFrame):
    last_ewm = data['bidPrice0'].ewm(alpha=.0001, min_periods=1000).mean().shift(-100)
    cur_ewm = data['bidPrice0'].ewm(alpha=.0001, min_periods=1000).mean()
    cur_exp_delta = (data['bidPrice0'].ewm(alpha=.0001, min_periods=1000).mean() - data['bidPrice0'].ewm(alpha=.0001, min_periods=1000).mean().shift(-1)).values
    last_exp_delta = (data['bidPrice0'].ewm(alpha=.0001, min_periods=1000).mean().shift(-1) - data['bidPrice0'].ewm(alpha=.0001, min_periods=1000).mean().shift(-2)).values

    return np.where((cur_exp_delta > 0) & (last_exp_delta < 0) & (cur_ewm < last_ewm), 1, np.where((cur_exp_delta < 0) & (last_exp_delta > 0) & (cur_ewm > last_ewm), -1, 0))

def single_ticker_rolling_exp_mean_delta_alpha_00005(data: pd.DataFrame):
    last_ewm = data['bidPrice0'].ewm(alpha=.00005, min_periods=1000).mean().shift(-100)
    cur_ewm = data['bidPrice0'].ewm(alpha=.00005, min_periods=1000).mean()
    cur_exp_delta = (data['bidPrice0'].ewm(alpha=.00005, min_periods=1000).mean() - data['bidPrice0'].ewm(alpha=.00005, min_periods=1000).mean().shift(-1)).values
    last_exp_delta = (data['bidPrice0'].ewm(alpha=.00005, min_periods=1000).mean().shift(-1) - data['bidPrice0'].ewm(alpha=.00005, min_periods=1000).mean().shift(-2)).values

    return np.where((cur_exp_delta > 0) & (last_exp_delta < 0) & (cur_ewm < last_ewm), 1, np.where((cur_exp_delta < 0) & (last_exp_delta > 0) & (cur_ewm > last_ewm), -1, 0))

def single_ticker_moving_average_crossover(data: pd.DataFrame, short_window: int = 50, long_window: int = 200, percentage_threshold: float = 0.005):
    short_mavg = data['bidPrice0'].rolling(window=short_window).mean()
    long_mavg = data['bidPrice0'].rolling(window=long_window).mean()
    mavg_diff = short_mavg - long_mavg

    return np.where(mavg_diff > percentage_threshold, 1, 
                    np.where(mavg_diff < -percentage_threshold * 2, -1,  # Reduce sell frequency by requiring a larger negative threshold
                             np.where((short_mavg.shift(1) <= long_mavg.shift(1)) & (short_mavg > long_mavg), 1, 
                                      np.where((short_mavg.shift(1) >= long_mavg.shift(1)) & (short_mavg < long_mavg), -1, 0))))

def single_ticker_volatility_breakout(data: pd.DataFrame, volatility_window: int = 20, breakout_multiplier: float = 2.0):
    rolling_std = data['bidPrice0'].rolling(window=volatility_window).std()
    upper_band = data['bidPrice0'].rolling(window=volatility_window).mean() + breakout_multiplier * rolling_std
    lower_band = data['bidPrice0'].rolling(window=volatility_window).mean() - breakout_multiplier * rolling_std

    return np.where(data['bidPrice0'] > upper_band, 1, 
                    np.where(data['bidPrice0'] < lower_band, -1, 0))

def single_ticker_mean_reversion(data: pd.DataFrame, lookback_period: int = 50, z_score_threshold: float = 2.0):
    rolling_mean = data['bidPrice0'].rolling(window=lookback_period).mean()
    rolling_std = data['bidPrice0'].rolling(window=lookback_period).std()
    z_score = (data['bidPrice0'] - rolling_mean) / rolling_std

    return np.where(z_score > z_score_threshold, -1, 
                    np.where(z_score < -z_score_threshold, 1, 0))

def single_ticker_rsi_strategy(data: pd.DataFrame, period: int = 14, overbought: float = 80, oversold: float = 20):
    delta = data['bidPrice0'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return np.where(rsi > overbought, -1, np.where(rsi < oversold, 1, 0))

def single_ticker_macd_strategy(data: pd.DataFrame, short_period: int = 12, long_period: int = 26, signal_period: int = 9):
    short_ema = data['bidPrice0'].ewm(span=short_period, adjust=False).mean()
    long_ema = data['bidPrice0'].ewm(span=long_period, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()

    return np.where(macd > signal + 0.5, 1, np.where(macd < signal - 0.5, -1, 0))

global_actions = {
    "single_ticker_rolling_mean_500_delta": Action(
        name="single_ticker_rolling_mean_500_delta",
        action_function=single_ticker_rolling_mean_500_delta
    ),
    "single_ticker_rolling_exp_mean_delta_alpha_00005": Action(
        name="single_ticker_rolling_exp_mean_delta_alpha_00005",
        action_function=single_ticker_rolling_exp_mean_delta_alpha_00005
    ),
    "single_ticker_rolling_exp_mean_delta_alpha_0001": Action(
        name="single_ticker_rolling_exp_mean_delta_alpha_0001",
        action_function=single_ticker_rolling_exp_mean_delta_alpha_0001
    ),
    "single_ticker_moving_average_crossover": Action(
        name="single_ticker_moving_average_crossover",
        action_function=single_ticker_moving_average_crossover
    ),
    "single_ticker_volatility_breakout": Action(
        name="single_ticker_volatility_breakout",
        action_function=single_ticker_volatility_breakout
    ),
    "single_ticker_mean_reversion": Action(
        name="single_ticker_mean_reversion",
        action_function=single_ticker_mean_reversion
    ),
    "single_ticker_rsi_strategy": Action(
        name="single_ticker_rsi_strategy",
        action_function=single_ticker_rsi_strategy
    ),
    "single_ticker_macd_strategy": Action(
        name="single_ticker_macd_strategy",
        action_function=single_ticker_macd_strategy
    )
}
