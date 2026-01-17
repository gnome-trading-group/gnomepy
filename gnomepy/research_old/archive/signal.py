from gnomepy.data.types import *
import pandas as pd
import numpy as np

class Signal:
    def __init__(self, name: str):
        self.name = name

class SimpleSignal(Signal):
    def __init__(self, name: str, pd_expression: str = None, np_expression: str = None):
        super().__init__(name)
        self.pd_expression = pd_expression if pd_expression is not None else ''
        self.np_expression = np_expression if np_expression is not None else ''

    def generate_signal(self, data: pd.DataFrame | np.ndarray, columns: list[str]) -> pd.DataFrame | np.ndarray:
        if isinstance(data, pd.DataFrame):
            result = data.copy()
            for column in columns:
                if column in data.columns:
                    new_column_name = f"{column}_{self.name}"
                    result[new_column_name] = eval(self.pd_expression.replace('data', f"data['{column}']"))
                else:
                    raise ValueError(f"Column '{column}' is not in the DataFrame")
            return result
        
        ## TODO: Implement this
        elif isinstance(data, np.ndarray):
            return eval(self.np_expression)
        else:
            raise TypeError("Data must be a pandas DataFrame or a numpy ndarray")

class CompoundSignal(Signal):
    def __init__(self, name: str, signals: list[tuple[Signal, str]], pd_expression: str = None):
        super().__init__(name)
        self.signals = signals
        self.pd_expression = pd_expression if pd_expression is not None else ''

    def generate_signal(self, data: pd.DataFrame, columns: dict) -> pd.DataFrame:

        if isinstance(data, pd.DataFrame):

            result = data.copy()
            
            # First generate simple signals
            for signal, dummy_column in self.signals:
                if columns[dummy_column] in data.columns:
                    new_column_name = f"{columns[dummy_column]}_{signal.name}"
                    result[new_column_name] = signal.generate_signal(data, [columns[dummy_column]])[new_column_name]
                else:
                    raise ValueError(f"Column '{columns[dummy_column]}' is not in the DataFrame")
            
            # Now generate expression
            if self.pd_expression:
                expression = self.pd_expression
                for key, value in columns.items():
                    expression = expression.replace(key, f"'{value}'")

                result[f"{self.name}"] = eval(expression)
                
            return result
        
        ## TODO: Implement this
        elif isinstance(data, np.ndarray):
            return eval(self.np_expression)
        else:
            raise TypeError("Data must be a pandas DataFrame or a numpy ndarray")

global_simple_signals = {
    "rolling_mean_30": SimpleSignal(
        name='rolling_mean_30',
        pd_expression="data.rolling(window=30).mean()",
    ),
    "rolling_mean_50": SimpleSignal(
        name='rolling_mean_50',
        pd_expression="data.rolling(window=50).mean()",
    ),
    "rolling_std_20": SimpleSignal(
        name='rolling_std_20',
        pd_expression="data.rolling(window=20).std()",
    ),
    "rolling_median_40": SimpleSignal(
        name='rolling_median_40',
        pd_expression="data.rolling(window=40).median()",
    ),
    "rolling_mean_10": SimpleSignal(
        name='rolling_mean_10',
        pd_expression="data.rolling(window=10).mean()",
    ),
    "rolling_mean_100": SimpleSignal(
        name='rolling_mean_100',
        pd_expression="data.rolling(window=100).mean()",
    ),
    "rolling_mean_500": SimpleSignal(
        name='rolling_mean_500',
        pd_expression="data.rolling(window=500).mean()",
    ),
    "rolling_mean_5000": SimpleSignal(
        name='rolling_mean_5000',
        pd_expression="data.rolling(window=5000).mean()",
    ),
    "rolling_mean_50000": SimpleSignal(
        name='rolling_mean_50000',
        pd_expression="data.rolling(window=50000).mean()",
    ),
    "rolling_mean_500_shifted": SimpleSignal(
        name='rolling_mean_500_shifted',
        pd_expression="data.rolling(window=500).mean().shift(-1)",
    )
}

globabl_compound_signals = {
    "single_ticker_rolling_mean_500_delta": CompoundSignal(
        name='single_ticker_rolling_mean_500_delta',
        signals=[
            (global_simple_signals['rolling_mean_500'], 'column0'),
            (global_simple_signals['rolling_mean_500_shifted'], 'column1')
        ],
        pd_expression="data[column0].rolling(window=500).mean() - data[column1].rolling(window=500).mean().shift(-1)",
    )
}
