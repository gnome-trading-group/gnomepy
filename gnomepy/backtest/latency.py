from abc import ABC, abstractmethod

import numpy as np


class LatencyModel(ABC):

    @abstractmethod
    def simulate(self) -> int:
        """Simulate the number of nanoseconds for an operation."""
        raise NotImplementedError


class GaussianLatency(LatencyModel):

    def __init__(self, mu: float, sigma:  float):
        self.mu = mu
        self.sigma = sigma

    def simulate(self) -> int:
        return np.random.normal(loc=self.mu, scale=self.sigma).astype(int)
