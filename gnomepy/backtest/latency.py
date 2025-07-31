from abc import ABC, abstractmethod
from typing import override

import numpy as np


class LatencyModel(ABC):

    @abstractmethod
    def simulate(self) -> int:
        """Simulate the number of nanoseconds for an operation."""
        ...

class StaticLatency(LatencyModel):

    def __init__(self, latency: int):
        self.latency = latency

    @override
    def simulate(self) -> int:
        return self.latency

class GaussianLatency(LatencyModel):

    def __init__(self, mu: float, sigma:  float):
        self.mu = mu
        self.sigma = sigma

    @override
    def simulate(self) -> int:
        return int(np.random.normal(loc=self.mu, scale=self.sigma))
