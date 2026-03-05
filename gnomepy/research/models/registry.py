"""
Directory-based model versioning for multi-type models.

Structure::

    {base_dir}/{model_type}/{model_name}/
    ├── registry.json        # Index of all versions
    ├── v1/
    │   ├── ...              # Model components
    │   └── metadata.json    # Training metadata
    ├── v2/
    │   ├── ...
    │   └── metadata.json
    └── ...
"""

import importlib
import json
import datetime
from pathlib import Path

import pandas as pd

from gnomepy.research.models.base import RegistrableModel


class ModelRegistry:
    """Register, load, and compare model versions across model types."""

    def __init__(self, base_dir: str = "./models"):
        self.base_dir = Path(base_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        model: RegistrableModel,
        metadata: dict,
    ) -> str:
        """Save a model as the next version.

        Parameters
        ----------
        model : RegistrableModel
            Trained model instance.  Its ``model_type`` and ``model_name``
            properties determine the directory.  Its ``save_to_dir`` method
            is called to persist artifacts.
        metadata : dict
            Training metadata (params, metrics, feature importance, etc.).

        Returns
        -------
        str
            Version string, e.g. ``"v3"``.
        """
        type_dir = self.base_dir / model.model_type / model.model_name
        registry_index = self._load_registry_index(type_dir)

        next_version_num = len(registry_index.get("versions", [])) + 1
        version = f"v{next_version_num}"

        version_dir = type_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Delegate artifact saving to the model itself
        model.save_to_dir(version_dir)

        # Store the fully-qualified class name so load() can reconstruct it
        fqn = f"{type(model).__module__}.{type(model).__qualname__}"

        # Attach version, timestamp, and class name to metadata
        metadata = {
            **metadata,
            "model_class": fqn,
            "version": version,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        with open(version_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Update registry index
        summary = {
            "version": version,
            "created_at": metadata["created_at"],
            "model_class": fqn,
        }
        # Copy all walk-forward metrics (excluding per_fold) into the summary
        wf = metadata.get("walk_forward_metrics", {})
        for k, v in wf.items():
            if k != "per_fold":
                summary[k] = v

        if "versions" not in registry_index:
            registry_index["versions"] = []
        registry_index["versions"].append(summary)

        self._save_registry_index(type_dir, registry_index)

        return version

    def load(
        self, model_type: str, model_name: str, version: str
    ) -> tuple[RegistrableModel, dict]:
        """Load a specific version's model and metadata.

        Parameters
        ----------
        model_type : str
            e.g. ``"directional"``
        model_name : str
            e.g. ``"listing_1"``
        version : str
            e.g. ``"v3"``

        Returns
        -------
        tuple[RegistrableModel, dict]
        """
        version_dir = self.base_dir / model_type / model_name / version
        if not version_dir.exists():
            raise FileNotFoundError(
                f"Version {version} not found for {model_type}/{model_name}"
            )

        with open(version_dir / "metadata.json") as f:
            metadata = json.load(f)

        fqn = metadata["model_class"]
        module_name, class_name = fqn.rsplit(".", 1)
        model_class = getattr(importlib.import_module(module_name), class_name)
        model = model_class.load_from_dir(version_dir, metadata)

        return model, metadata

    def load_latest(
        self, model_type: str, model_name: str
    ) -> tuple[RegistrableModel, dict]:
        """Load the most recent version."""
        versions = self.list_versions(model_type, model_name)
        if not versions:
            raise FileNotFoundError(
                f"No versions found for {model_type}/{model_name}"
            )
        latest = versions[-1]["version"]
        return self.load(model_type, model_name, latest)

    def load_best(
        self, model_type: str, model_name: str, metric: str = "mean_auc"
    ) -> tuple[RegistrableModel, dict]:
        """Load the version with the best out-of-sample metric.

        Parameters
        ----------
        model_type : str
        model_name : str
        metric : str
            Exact key in the registry summary, e.g. ``"mean_auc"``,
            ``"mean_rmse"``.  Higher values are preferred; fall back to
            latest if no version has the metric.

        Returns
        -------
        tuple[RegistrableModel, dict]
        """
        versions = self.list_versions(model_type, model_name)
        if not versions:
            raise FileNotFoundError(
                f"No versions found for {model_type}/{model_name}"
            )

        best = max(
            (v for v in versions if v.get(metric) is not None),
            key=lambda v: v[metric],
            default=None,
        )
        if best is None:
            return self.load_latest(model_type, model_name)

        return self.load(model_type, model_name, best["version"])

    def list_versions(self, model_type: str, model_name: str) -> list[dict]:
        """List all versions with summary metrics."""
        type_dir = self.base_dir / model_type / model_name
        registry_index = self._load_registry_index(type_dir)
        return registry_index.get("versions", [])

    def compare(
        self,
        model_type: str,
        model_name: str,
        versions: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compare metrics across versions.

        Parameters
        ----------
        model_type : str
        model_name : str
        versions : list[str] or None
            Specific versions to compare. If ``None``, compares all.

        Returns
        -------
        pd.DataFrame
            One row per version with metric columns.
        """
        all_versions = self.list_versions(model_type, model_name)
        if versions is not None:
            all_versions = [v for v in all_versions if v["version"] in versions]
        return pd.DataFrame(all_versions).set_index("version")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_registry_index(self, type_dir: Path) -> dict:
        index_path = type_dir / "registry.json"
        if index_path.exists():
            with open(index_path) as f:
                return json.load(f)
        return {}

    def _save_registry_index(self, type_dir: Path, index: dict) -> None:
        type_dir.mkdir(parents=True, exist_ok=True)
        with open(type_dir / "registry.json", "w") as f:
            json.dump(index, f, indent=2, default=str)
