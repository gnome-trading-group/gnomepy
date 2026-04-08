from __future__ import annotations

import os
from pathlib import Path

GNOME_JARS_ENV = "GNOME_JARS"
GNOME_ROOT_ENV = "GNOME_ROOT"
GNOME_USER_JARS_ENV = "GNOME_USER_JARS"
TARGET_JAR = "gnome-backtest"

def discover_classpath(
    gnome_root: str | Path | None = None,
    extra_jars: list[str] | None = None,
) -> list[str]:
    """Discover JARs to put on the JVM classpath.

    Resolution order for the core gnome JARs:
    1. GNOME_JARS env var — explicit comma-separated JAR paths
    2. GNOME_ROOT env var or gnome_root param — scan target/ dirs for uber JARs
    3. Fallback — walk up from this package looking for sibling Maven projects

    Additional user JARs (e.g. compiled strategy code) are merged in from:
    - extra_jars argument (highest priority)
    - GNOME_USER_JARS env var (comma-separated)
    """
    core: list[str] = []

    # 1. Explicit JAR list
    explicit = os.environ.get(GNOME_JARS_ENV)
    if explicit:
        core = [j.strip() for j in explicit.split(",") if j.strip()]
        missing = [j for j in core if not Path(j).exists()]
        if missing:
            raise FileNotFoundError(f"JARs specified in {GNOME_JARS_ENV} not found: {missing}")
    else:
        # 2. Root directory scan
        root = gnome_root or os.environ.get(GNOME_ROOT_ENV)
        if root:
            core = _scan_root(Path(root))
        else:
            # 3. Fallback: walk up from this file to find sibling projects
            current = Path(__file__).resolve()
            for parent in current.parents:
                if (parent / "gnome-backtest").is_dir():
                    core = _scan_root(parent)
                    break
            if not core:
                raise FileNotFoundError(
                    f"Cannot discover GNOME JARs. Set {GNOME_JARS_ENV} or {GNOME_ROOT_ENV} "
                    "environment variable, or pass gnome_root to ensure_jvm_started()."
                )

    # Merge user JARs (env first, then explicit args — args win on dedup)
    user: list[str] = []
    env_user = os.environ.get(GNOME_USER_JARS_ENV)
    if env_user:
        user.extend(j.strip() for j in env_user.split(",") if j.strip())
    if extra_jars:
        user.extend(extra_jars)

    missing = [j for j in user if not Path(j).exists()]
    if missing:
        raise FileNotFoundError(f"User JARs not found: {missing}")

    # Dedup while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for j in core + user:
        if j not in seen:
            seen.add(j)
            out.append(j)
    return out


def _scan_root(root: Path) -> list[str]:
    """Scan a GNOME root directory for target JARs."""
    jars = []
    target = root / TARGET_JAR / "target"
    if target.is_dir():
        for jar in sorted(target.glob("*-all.jar"), reverse=True):
            jars.append(str(jar))
            break  # take the most recent one per project
    else:
        target = root / TARGET_JAR / "target"
        if target.is_dir():
            for jar in sorted(target.glob("*.jar"), reverse=True):
                if "original" not in jar.name and "sources" not in jar.name:
                    jars.append(str(jar))
                    break
    if not jars:
        raise FileNotFoundError(
            f"No uber JARs found under {root}. "
            f"Build the Java projects first: cd {TARGET_JAR} && mvn package -DskipTests"
        )
    return jars
