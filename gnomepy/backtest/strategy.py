from gnomepy.data.types import *
from gnomepy.backtest.signal import *
from gnomepy.backtest.action import *
import pandas as pd
import numpy as np

class Strategy:
    def __init__(self, name: str, signals: list[Signal], action: Action):
        self.name = name
        self.signals = signals
        self.action = action
        
    def compute_signals(self, data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        if isinstance(data, pd.DataFrame):
            for signal in self.signals:
                data[signal.name] = eval(signal.pd_expression)
        elif isinstance(data, np.ndarray):
            for signal in self.signals:
                exec(f"data = {signal.np_expression}")
        return data

    def execute(self, data: pd.DataFrame):
        computed_data = self.compute_signals(data)
        return self.action.apply(computed_data)