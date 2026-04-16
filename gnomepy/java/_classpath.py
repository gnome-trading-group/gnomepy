from __future__ import annotations

import os
from pathlib import Path

GNOME_JARS_ENV = "GNOME_JARS"
GNOME_ROOT_ENV = "GNOME_ROOT"
TARGET_JAR = "gnome-backtest"


def discover_classpath(gnome_root: str | Path | None = None) -> list[str]:
    """Discover JARs to put on the JVM classpath.

    Resolution order:
    1. GNOME_JARS env var — explicit comma-separated JAR paths
    2. GNOME_ROOT env var or gnome_root param — scan target/ dirs for uber JARs
    3. Fallback — walk up from this package looking for sibling Maven projects
    """
    # 1. Explicit JAR list
    explicit = os.environ.get(GNOME_JARS_ENV)
    if explicit:
        core = [j.strip() for j in explicit.split(",") if j.strip()]
        missing = [j for j in core if not Path(j).exists()]
        if missing:
            raise FileNotFoundError(f"JARs specified in {GNOME_JARS_ENV} not found: {missing}")
        return core

    # 2. Root directory scan
    root = gnome_root or os.environ.get(GNOME_ROOT_ENV)
    if root:
        return _scan_root(Path(root))

    # 3. Fallback: walk up from this file to find sibling projects
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "gnome-backtest").is_dir():
            return _scan_root(parent)

    raise FileNotFoundError(
        f"Cannot discover GNOME JARs. Set {GNOME_JARS_ENV} or {GNOME_ROOT_ENV} "
        "environment variable, or pass gnome_root to ensure_jvm_started()."
    )


def _scan_root(root: Path) -> list[str]:
    """Scan a GNOME root directory for the gnome-backtest uber JAR."""
    target = root / TARGET_JAR / "target"
    if target.is_dir():
        for jar in sorted(target.glob("*-all.jar"), reverse=True):
            return [str(jar)]
        for jar in sorted(target.glob("*.jar"), reverse=True):
            if "original" not in jar.name and "sources" not in jar.name:
                return [str(jar)]
    raise FileNotFoundError(
        f"No uber JAR found under {root / TARGET_JAR}. "
        f"Build first: cd {TARGET_JAR} && mvn package -DskipTests"
    )
