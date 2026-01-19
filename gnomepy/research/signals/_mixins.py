import numpy as np
import pandas as pd


class PricingMixin:

    def calculate_spread(self, bid_price: float | np.ndarray, ask_price: float | np.ndarray) -> float | np.ndarray:
        return ask_price - bid_price

    def calculate_mid_price(self, bid_price: float | np.ndarray, ask_price: float | np.ndarray) -> float | np.ndarray:
        return (bid_price + ask_price) / 2.0


class VolatilityMixin:

    def calculate_volatility(self, prices: np.ndarray, method: str = "simple", /, **kwargs) -> float:
        if method == "ewm":
            return self.calculate_ewm_volatility(prices, **kwargs)
        elif method == "simple":
            return self.calculate_simple_volatility(prices, **kwargs)
        else:
            raise ValueError(f"Invalid method: {method}")

    def calculate_simple_volatility(self, prices: np.ndarray, window: int = 100) -> float:
        if len(prices) < 2:
            return 0.0

        return prices[-window:].std()

    def calculate_ewm_volatility(self, prices: np.ndarray, window: int = 100) -> float:
        if len(prices) < 2:
            return 0.0

        log_returns = np.diff(np.log(prices))
        if len(log_returns) == 0:
            return 0.0

        return pd.Series(log_returns).ewm(span=window).std().iloc[-1]


class InventoryMixin:

    def calculate_inventory(self, inventory: np.ndarray, levels: int = 1) -> float:
        return inventory[:levels].sum()

    def calculate_imbalance(self, inventory: np.ndarray, levels: int = 1) -> float:
        bid_inventory = inventory[:levels].sum()
        ask_inventory = inventory[:levels].sum()
        if bid_inventory + ask_inventory == 0:
            return 0.0
        return bid_inventory / (bid_inventory + ask_inventory)
