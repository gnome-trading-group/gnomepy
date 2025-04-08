from typing import Callable
import pandas as pd
import numpy as np

class Action:
    def __init__(self, action_function: Callable):
        self.action_function = action_function

    def apply(self, data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        if isinstance(data, pd.DataFrame):
            data['action'] = self.action_function(data)
        elif isinstance(data, np.ndarray):
            action_column = self.action_function(data)
            data = np.column_stack((data, action_column))
        return data
