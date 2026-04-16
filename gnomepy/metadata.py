from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BacktestMetadata:
    """Metadata for a single backtest run.

    Persisted as ``metadata.json`` alongside parquet output files.
    """

    backtest_id: str

    # Auto-populated
    created_at: str = field(default_factory=_now_utc_iso)

    # Timing
    start_date: str | None = None
    end_date: str | None = None
    wall_time_seconds: float | None = None
    event_count: int | None = None

    # Strategy
    strategy: str | None = None
    strategy_args: dict[str, Any] | None = None

    # Configuration
    config_path: str | None = None
    config: dict[str, Any] | None = None
    preset_name: str | None = None

    # Provenance
    gnomepy_version: str | None = None
    gnomepy_research_version: str | None = None
    gnomepy_research_commit: str | None = None

    # Persistence
    data_scaled: bool = True

    # Extensible
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict, omitting None values."""
        result: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if v is not None:
                result[f.name] = v
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BacktestMetadata:
        """Deserialize from a dict. Unknown keys are stored in ``extra``."""
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in known}
        unknown = {k: v for k, v in d.items() if k not in known}
        if "extra" in kwargs:
            unknown.update(kwargs["extra"])
        obj = cls(**{k: v for k, v in kwargs.items() if k != "extra"})
        obj.extra.update(unknown)
        return obj

    def save(self, directory: str | Path) -> None:
        """Write ``metadata.json`` to the given directory."""
        p = Path(directory) / "metadata.json"
        p.write_text(json.dumps(self.to_dict(), indent=2, default=str) + "\n")

    @classmethod
    def load(cls, directory: str | Path) -> BacktestMetadata | None:
        """Load ``metadata.json`` from a directory. Returns None if not found."""
        p = Path(directory) / "metadata.json"
        if not p.exists():
            return None
        return cls.from_dict(json.loads(p.read_text()))
