from gnomepy.data.types import *
from gnomepy.backtest.signal import *
from gnomepy.backtest.action import *
import pandas as pd
import numpy as np

class Strategy:
    def __init__(self, name: str, signals: list[tuple[Signal, list[str]]], action: Action):
        self.name = name
        self.signals = signals
        self.action = action
        
    def compute_signals(self, data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        if isinstance(data, pd.DataFrame):
            for signal in self.signals:                
                if isinstance(signal['signal'], SimpleSignal):
                    data = signal['signal'].generate_signal(data=data, columns=signal['columns'])
                
                # TODO: We need to figure out how we are dealing with compound signals here
                elif isinstance(signal['signal'], CompoundSignal):
                    pass

        elif isinstance(data, np.ndarray):
            for signal in self.signals:
                exec(f"data = {signal.np_expression}")
        return data

    def execute(self, data: pd.DataFrame):
        computed_data = self.compute_signals(data)

        # TODO: Apply the actual action function
        # return self.action.apply(computed_data)

        # This currently just returns signals
        return computed_data