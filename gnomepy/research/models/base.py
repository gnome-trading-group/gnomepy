"""Abstract base class for registrable models."""

from abc import ABC, abstractmethod
from pathlib import Path


class RegistrableModel(ABC):
    """Mixin ABC that any model stored in ModelRegistry must implement."""

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Model family, e.g. 'directional', 'volatility'."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Parameter-based name, e.g. 'listing_1' or 'realized_w100_h20'."""

    @abstractmethod
    def save_to_dir(self, path: Path) -> None:
        """Save model artifacts to a version directory."""

    @classmethod
    @abstractmethod
    def load_from_dir(cls, path: Path, metadata: dict | None = None) -> "RegistrableModel":
        """Load model from a version directory.

        Parameters
        ----------
        path : Path
            Version directory (e.g. ``./models/1/directional/v3``).
        metadata : dict or None
            Already-loaded metadata dict from the registry.  If ``None``
            (e.g. path-based loading), implementations should read
            ``metadata.json`` themselves.
        """
