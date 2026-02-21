"""
Directory-based model versioning for LightGBM directional models.

Structure::

    {base_dir}/{listing_id}/
    ├── registry.json        # Index of all versions
    ├── v1/
    │   ├── model.txt        # LightGBM native model
    │   └── metadata.json    # Training metadata
    ├── v2/
    │   ├── model.txt
    │   └── metadata.json
    └── ...
"""

import json
import datetime
from pathlib import Path

import lightgbm as lgb
import pandas as pd


class ModelRegistry:
    """Register, load, and compare LightGBM model versions."""

    def __init__(self, base_dir: str = "./models"):
        self.base_dir = Path(base_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        listing_id: int,
        model: lgb.Booster,
        metadata: dict,
    ) -> str:
        """Save a model as the next version.

        Parameters
        ----------
        listing_id : int
            The listing this model was trained for.
        model : lgb.Booster
            Trained LightGBM model.
        metadata : dict
            Training metadata (params, metrics, feature importance, etc.).

        Returns
        -------
        str
            Version string, e.g. ``"v3"``.
        """
        listing_dir = self.base_dir / str(listing_id)
        registry_index = self._load_registry_index(listing_dir)

        next_version_num = len(registry_index.get("versions", [])) + 1
        version = f"v{next_version_num}"

        version_dir = listing_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save model in LightGBM native text format
        model.save_model(str(version_dir / "model.txt"))

        # Attach version & timestamp to metadata
        metadata = {
            **metadata,
            "version": version,
            "listing_id": listing_id,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        with open(version_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Update registry index
        summary = {
            "version": version,
            "created_at": metadata["created_at"],
        }
        # Copy select metrics into the summary for quick comparison
        wf = metadata.get("walk_forward_metrics", {})
        if wf:
            summary["mean_auc"] = wf.get("mean_auc")
            summary["mean_accuracy"] = wf.get("mean_accuracy")

        if "versions" not in registry_index:
            registry_index["versions"] = []
        registry_index["versions"].append(summary)

        self._save_registry_index(listing_dir, registry_index)

        return version

    def load(self, listing_id: int, version: str) -> tuple[lgb.Booster, dict]:
        """Load a specific version's model and metadata.

        Parameters
        ----------
        listing_id : int
        version : str
            e.g. ``"v3"``

        Returns
        -------
        tuple[lgb.Booster, dict]
        """
        version_dir = self.base_dir / str(listing_id) / version
        if not version_dir.exists():
            raise FileNotFoundError(f"Version {version} not found for listing {listing_id}")

        model = lgb.Booster(model_file=str(version_dir / "model.txt"))
        with open(version_dir / "metadata.json") as f:
            metadata = json.load(f)

        return model, metadata

    def load_latest(self, listing_id: int) -> tuple[lgb.Booster, dict]:
        """Load the most recent version."""
        versions = self.list_versions(listing_id)
        if not versions:
            raise FileNotFoundError(f"No versions found for listing {listing_id}")
        latest = versions[-1]["version"]
        return self.load(listing_id, latest)

    def load_best(
        self, listing_id: int, metric: str = "auc"
    ) -> tuple[lgb.Booster, dict]:
        """Load the version with the best out-of-sample metric.

        Parameters
        ----------
        listing_id : int
        metric : str
            Metric key inside ``walk_forward_metrics``, e.g. ``"auc"`` looks
            for ``mean_auc`` in the registry index.

        Returns
        -------
        tuple[lgb.Booster, dict]
        """
        versions = self.list_versions(listing_id)
        if not versions:
            raise FileNotFoundError(f"No versions found for listing {listing_id}")

        key = f"mean_{metric}"
        best = max(
            (v for v in versions if v.get(key) is not None),
            key=lambda v: v[key],
            default=None,
        )
        if best is None:
            # Fall back to latest if no metric available
            return self.load_latest(listing_id)

        return self.load(listing_id, best["version"])

    def list_versions(self, listing_id: int) -> list[dict]:
        """List all versions with summary metrics."""
        listing_dir = self.base_dir / str(listing_id)
        registry_index = self._load_registry_index(listing_dir)
        return registry_index.get("versions", [])

    def compare(
        self, listing_id: int, versions: list[str] | None = None
    ) -> pd.DataFrame:
        """Compare metrics across versions.

        Parameters
        ----------
        listing_id : int
        versions : list[str] or None
            Specific versions to compare. If ``None``, compares all.

        Returns
        -------
        pd.DataFrame
            One row per version with metric columns.
        """
        all_versions = self.list_versions(listing_id)
        if versions is not None:
            all_versions = [v for v in all_versions if v["version"] in versions]
        return pd.DataFrame(all_versions).set_index("version")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_registry_index(self, listing_dir: Path) -> dict:
        index_path = listing_dir / "registry.json"
        if index_path.exists():
            with open(index_path) as f:
                return json.load(f)
        return {}

    def _save_registry_index(self, listing_dir: Path, index: dict) -> None:
        listing_dir.mkdir(parents=True, exist_ok=True)
        with open(listing_dir / "registry.json", "w") as f:
            json.dump(index, f, indent=2, default=str)
