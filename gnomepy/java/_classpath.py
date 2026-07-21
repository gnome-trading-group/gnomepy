from __future__ import annotations

import os
from pathlib import Path

GNOME_JARS_ENV = "GNOME_JARS"
GNOME_ROOT_ENV = "GNOME_ROOT"


def discover_classpath(jar_name: str, gnome_root: str | Path | None = None) -> list[str]:
    """Discover the uber JAR for a gnome Maven project.

    Resolution order:
    1. GNOME_JARS env var — explicit comma-separated JAR paths
    2. GNOME_ROOT env var or gnome_root param — scan {jar_name}/target/
    3. Fallback — walk up from this file looking for a sibling {jar_name} directory
    """
    explicit = os.environ.get(GNOME_JARS_ENV)
    if explicit:
        core = [j.strip() for j in explicit.split(",") if j.strip()]
        missing = [j for j in core if not Path(j).exists()]
        if missing:
            raise FileNotFoundError(f"JARs specified in {GNOME_JARS_ENV} not found: {missing}")
        return core

    root = gnome_root or os.environ.get(GNOME_ROOT_ENV)
    if root:
        return _scan_project(Path(root), jar_name)

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / jar_name).is_dir():
            return _scan_project(parent, jar_name)

    raise FileNotFoundError(
        f"Cannot discover {jar_name} JAR. Set {GNOME_JARS_ENV} or {GNOME_ROOT_ENV}, "
        f"or ensure {jar_name} is in a sibling directory. "
        f"Build first: cd {jar_name} && mvn package -DskipTests"
    )


def _scan_project(root: Path, jar_name: str) -> list[str]:
    target = root / jar_name / "target"
    if target.is_dir():
        for jar in sorted(target.glob("*-all.jar"), reverse=True):
            return [str(jar)]
        for jar in sorted(target.glob("*.jar"), reverse=True):
            if "original" not in jar.name and "sources" not in jar.name:
                return [str(jar)]
    raise FileNotFoundError(
        f"No uber JAR found under {root / jar_name}. "
        f"Build first: cd {jar_name} && mvn package -DskipTests"
    )
