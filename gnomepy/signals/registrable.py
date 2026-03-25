from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Registrable(ABC):
    """Base class for saveable/loadable signal models.

    Models save their parameters and state to a directory (format is
    up to the model). The SignalRegistry handles syncing directories
    to/from S3.
    """

    @abstractmethod
    def save_model(self, directory: Path):
        """Save parameters + state to the given directory."""
        ...

    @abstractmethod
    def load_model(self, directory: Path):
        """Load parameters + state from the given directory."""
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Model name (e.g., 'ewma-volatility')."""
        ...

    @abstractmethod
    def get_version(self) -> str:
        """Model version (e.g., 'v1.0')."""
        ...
