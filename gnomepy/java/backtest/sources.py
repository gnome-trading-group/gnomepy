"""What a backtest runs: the gnome-backtest engine jar, and the gnome-strategies classes a Java strategy comes from.

A Java strategy runs from a gnome-strategies commit (``strategy_commit``) or a local checkout (``strategy_source``),
compiled against the engine as a session compiles it against the orchestrator. The engine is the latest one on the
classpath, or a released one (``engine_version``) to rerun an old result as it ran. Every run records both, and warns
when the engine's runtime or OMS differs from what the strategy commit's pinned orchestrator runs.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import boto3
import requests

from gnomepy.java._classpath import GNOME_JARS_ENV, discover_classpath

GITHUB_API = "https://api.github.com"
GITHUB_MAVEN = "https://maven.pkg.github.com"
GITHUB_OWNER = "gnome-trading-group"
# The token sessions clone their strategy commit with, which also reads GitHub Packages.
GH_TOKEN_ENV = "GH_TOKEN"
GH_TOKEN_SECRET = "gnomepy/gh-token"
GH_TOKEN_REGION = "us-east-1"
GITHUB_TIMEOUT_SECONDS = 60

STRATEGIES_REPO = "gnome-strategies"
STRATEGIES_MODULE = "strategies"
ORCHESTRATOR_REPO = "gnome-orchestrator"
BACKTEST_REPO = "gnome-backtest"
GROUP_PATH = "group/gnometrading"
# Sessions compile with the same flags (gnome-orchestrator start-session.sh): StrategyFactory matches args to
# constructor parameters by name.
JAVAC_FLAGS = ["--release", "17", "-parameters", "-nowarn"]
RELEASE_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")

CACHE_ROOT = Path.home() / ".cache" / "gnome"

_token: str | None = None


class StrategySourceError(Exception):
    """A strategy commit, checkout or engine version that can't be run, as given."""


@dataclass
class BacktestSources:
    """The jars and classes a backtest's JVM starts with, and what they are, for its results."""

    classpath: list[str]
    engine_jar: str | None = None
    engine_jar_hash: str | None = None
    engine_version: str | None = None
    runtime_version: str | None = None
    oms_version: str | None = None
    # "commit" or "local"; None when the strategy comes from the engine jar (or is Python).
    strategy_source: str | None = None
    strategy_commit: str | None = None
    # A local checkout with uncommitted changes under strategies/: its run is not strategy_commit's.
    strategy_dirty: bool | None = None
    platform_orchestrator: str | None = None
    warnings: list[str] = field(default_factory=list)


def github_token() -> str:
    global _token
    if _token:
        return _token
    token = os.environ.get(GH_TOKEN_ENV, "").strip()
    if not token:
        raw = boto3.client("secretsmanager", region_name=GH_TOKEN_REGION).get_secret_value(
            SecretId=GH_TOKEN_SECRET
        )["SecretString"]
        try:
            token = json.loads(raw)["token"]
        except (json.JSONDecodeError, KeyError, TypeError):
            token = raw.strip()
    if not token:
        raise StrategySourceError(f"no GitHub token: set {GH_TOKEN_ENV} or store one in {GH_TOKEN_SECRET}")
    _token = token
    return _token


def _github(path: str, accept: str, stream: bool = False) -> requests.Response:
    return requests.get(
        f"{GITHUB_API}/repos/{GITHUB_OWNER}/{path}",
        headers={
            "Authorization": f"Bearer {github_token()}",
            "Accept": accept,
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=GITHUB_TIMEOUT_SECONDS,
        stream=stream,
    )


def resolve_strategy_commit(ref: str) -> str:
    """The full SHA ``ref`` (a branch, tag or SHA of gnome-strategies) names now."""
    res = _github(f"{STRATEGIES_REPO}/commits/{quote(ref, safe='')}", "application/vnd.github.sha")
    if res.status_code in (404, 422):
        raise StrategySourceError(f"{STRATEGIES_REPO} has no commit {ref}")
    res.raise_for_status()
    return res.text.strip()


def pinned_orchestrator(pom: str) -> str | None:
    """The orchestrator release a strategies/pom.xml pins (``platform.orchestrator``)."""
    match = re.search(r"<platform\.orchestrator>\s*([^<\s]+)\s*</platform\.orchestrator>", pom)
    return match.group(1) if match else None


def _dependency_version(pom: str, artifact: str) -> str | None:
    match = re.search(
        rf"<artifactId>{re.escape(artifact)}</artifactId>\s*<version>\s*([^<\s]+)\s*</version>", pom
    )
    return match.group(1) if match else None


def jar_versions(jar: str | Path) -> dict[str, str]:
    """Each gnome-trading-group artifact's version inside ``jar``, from the pom.properties Maven packs."""
    versions: dict[str, str] = {}
    prefix = f"META-INF/maven/{GROUP_PATH.replace('/', '.')}/"
    with zipfile.ZipFile(jar) as zf:
        for name in zf.namelist():
            if not (name.startswith(prefix) and name.endswith("/pom.properties")):
                continue
            props = {}
            for line in zf.read(name).decode().splitlines():
                if "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    props[key.strip()] = value.strip()
            if "artifactId" in props and "version" in props:
                versions[props["artifactId"]] = props["version"]
    return versions


def _jar_hash(jar: str | Path) -> str:
    digest = hashlib.sha256()
    with open(jar, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()[:16]}"


def _replace_dir(staged: Path, final: Path) -> None:
    # Two runs filling the same cache entry at once both stage; the first rename wins and the second's copy is
    # identical, so it's dropped.
    final.parent.mkdir(parents=True, exist_ok=True)
    try:
        staged.rename(final)
    except OSError:
        if not final.is_dir():
            raise
        shutil.rmtree(staged, ignore_errors=True)


def engine_jars(engine_version: str | None) -> list[str]:
    """The engine's jars: a released gnome-backtest's ``-all`` jar, or the one on the default classpath."""
    if not engine_version:
        return discover_classpath(BACKTEST_REPO)
    if os.environ.get(GNOME_JARS_ENV):
        raise StrategySourceError(f"--engine-version and {GNOME_JARS_ENV} both name an engine; pass one")
    if not RELEASE_VERSION.match(engine_version):
        raise StrategySourceError(f"engine version {engine_version} is not a release (X.Y.Z)")
    name = f"{BACKTEST_REPO}-{engine_version}-all.jar"
    jar = CACHE_ROOT / "engines" / name
    if jar.is_file():
        return [str(jar)]
    url = f"{GITHUB_MAVEN}/{GITHUB_OWNER}/{BACKTEST_REPO}/{GROUP_PATH}/{BACKTEST_REPO}/{engine_version}/{name}"
    res = requests.get(url, auth=("gnome", github_token()), timeout=GITHUB_TIMEOUT_SECONDS, stream=True)
    if res.status_code == 404:
        raise StrategySourceError(f"{BACKTEST_REPO} {engine_version} has no published {name}")
    res.raise_for_status()
    jar.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=jar.parent, suffix=".part", delete=False) as tmp:
        for chunk in res.iter_content(1 << 20):
            tmp.write(chunk)
    os.replace(tmp.name, jar)
    return [str(jar)]


def _fetch_strategies(sha: str) -> Path:
    """gnome-strategies' strategies/ module at ``sha``, cached: a commit's tree never changes."""
    final = CACHE_ROOT / "strategies" / sha / "source"
    if (final / "pom.xml").is_file():
        return final
    res = _github(f"{STRATEGIES_REPO}/tarball/{sha}", "application/vnd.github+json", stream=True)
    res.raise_for_status()
    final.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(dir=final.parent, prefix="source."))
    with tarfile.open(fileobj=res.raw, mode="r|gz") as tar:
        for member in tar:
            # Members sit under one "<owner>-<repo>-<sha>/" directory.
            parts = Path(member.name).parts
            if len(parts) < 3 or parts[1] != STRATEGIES_MODULE:
                continue
            member.name = str(Path(*parts[2:]))
            tar.extract(member, staged, filter="data")
    if not (staged / "pom.xml").is_file():
        shutil.rmtree(staged, ignore_errors=True)
        raise StrategySourceError(f"{STRATEGIES_REPO} {sha[:10]} has no {STRATEGIES_MODULE}/pom.xml")
    _replace_dir(staged, final)
    return final


def _javac() -> str:
    java_home = os.environ.get("JAVA_HOME")
    if java_home and (Path(java_home) / "bin" / "javac").is_file():
        return str(Path(java_home) / "bin" / "javac")
    found = shutil.which("javac")
    if not found:
        raise StrategySourceError("javac not found: a Java strategy is compiled for its backtest, so install a JDK")
    return found


def compile_strategies(module: Path, classpath: list[str], out: Path) -> None:
    """Compile a strategies/ module's sources into ``out``, with its resources, against ``classpath``."""
    sources = sorted(str(p) for p in (module / "src" / "main" / "java").rglob("*.java"))
    if not sources:
        raise StrategySourceError(f"{module} has no Java sources under src/main/java")
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as argfile:
        argfile.write("\n".join(f'"{s}"' for s in sources) + "\n")
    try:
        proc = subprocess.run(
            [_javac(), *JAVAC_FLAGS, "-cp", os.pathsep.join(classpath), "-d", str(out), f"@{argfile.name}"],
            capture_output=True,
            text=True,
        )
    finally:
        os.unlink(argfile.name)
    if proc.returncode != 0:
        raise StrategySourceError(f"compiling {module} failed:\n{proc.stdout}{proc.stderr}")
    resources = module / "src" / "main" / "resources"
    if resources.is_dir():
        shutil.copytree(resources, out, dirs_exist_ok=True)


def _strategy_commit_classes(sha: str, engine: list[str], engine_hash: str) -> tuple[Path, str | None]:
    module = _fetch_strategies(sha)
    # Keyed by the engine too: classes compiled against one engine can fail to link against another.
    final = CACHE_ROOT / "strategies" / sha / engine_hash.removeprefix("sha256:") / "classes"
    if not final.is_dir():
        final.parent.mkdir(parents=True, exist_ok=True)
        staged = Path(tempfile.mkdtemp(dir=final.parent, prefix="classes."))
        try:
            compile_strategies(module, engine, staged)
        except Exception:
            shutil.rmtree(staged, ignore_errors=True)
            raise
        _replace_dir(staged, final)
    return final, pinned_orchestrator((module / "pom.xml").read_text())


def local_strategy_module(path: str | Path) -> Path:
    """The strategies/ module of a gnome-strategies checkout, given the checkout or the module itself."""
    path = Path(path).expanduser().resolve()
    for candidate in (path / STRATEGIES_MODULE, path):
        if (candidate / "pom.xml").is_file() and (candidate / "src" / "main" / "java").is_dir():
            return candidate
    raise StrategySourceError(f"{path} is not a gnome-strategies checkout or its {STRATEGIES_MODULE}/ module")


def _git(module: Path, *args: str) -> str | None:
    proc = subprocess.run(["git", "-C", str(module), *args], capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else None


def _local_strategy_classes(path: str | Path, engine: list[str]) -> tuple[Path, str | None, str | None, bool | None]:
    module = local_strategy_module(path)
    # Compiled fresh every run: a cache keyed on anything but every source's content would hand back stale classes.
    out = Path(tempfile.mkdtemp(prefix="gnome-strategies-local-"))
    # Registered before the JVM starts, so it runs after the JVM's own exit hook has shut it down.
    atexit.register(shutil.rmtree, out, True)
    compile_strategies(module, engine, out)
    head = _git(module, "rev-parse", "HEAD")
    status = _git(module, "status", "--porcelain", "--", ".")
    dirty = None if status is None else bool(status)
    return out, head, pinned_orchestrator((module / "pom.xml").read_text()), dirty


def orchestrator_versions(version: str) -> dict[str, str | None]:
    """The gnome-strategies (runtime) and gnome-oms versions an orchestrator release runs, from its pom at its tag."""
    cached = CACHE_ROOT / "orchestrator" / f"{version}.json"
    if cached.is_file():
        return json.loads(cached.read_text())
    tag = f"{ORCHESTRATOR_REPO}-{version}"
    res = _github(f"{ORCHESTRATOR_REPO}/contents/pom.xml?ref={quote(tag, safe='')}", "application/vnd.github.raw")
    if res.status_code == 404:
        raise StrategySourceError(f"{ORCHESTRATOR_REPO} has no tag {tag}")
    res.raise_for_status()
    versions = {
        "runtime": _dependency_version(res.text, "gnome-strategies"),
        "oms": _dependency_version(res.text, "gnome-oms"),
    }
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(versions))
    return versions


def drift_warning(sources: BacktestSources, pinned: dict[str, str | None]) -> str | None:
    """Where the engine's runtime or OMS differs from the pinned orchestrator's; None when they agree."""
    differences = [
        f"{name} {mine} (orchestrator {sources.platform_orchestrator} runs {theirs})"
        for name, mine, theirs in (
            ("gnome-strategies", sources.runtime_version, pinned.get("runtime")),
            ("gnome-oms", sources.oms_version, pinned.get("oms")),
        )
        if mine != theirs
    ]
    if not differences:
        return None
    return (
        f"engine {BACKTEST_REPO} {sources.engine_version} differs from what the strategy's sessions run: "
        + "; ".join(differences)
    )


def describe_engine(classpath: list[str]) -> BacktestSources:
    """The engine on ``classpath``: its first jar holding gnome-backtest."""
    sources = BacktestSources(classpath=list(classpath))
    for entry in classpath:
        if not entry.endswith(".jar") or not Path(entry).is_file():
            continue
        versions = jar_versions(entry)
        if BACKTEST_REPO not in versions:
            continue
        sources.engine_jar = entry
        sources.engine_jar_hash = _jar_hash(entry)
        sources.engine_version = versions[BACKTEST_REPO]
        sources.runtime_version = versions.get("gnome-strategies")
        sources.oms_version = versions.get("gnome-oms")
        break
    return sources


def prepare_sources(
    strategy_commit: str | None = None,
    strategy_source: str | Path | None = None,
    engine_version: str | None = None,
) -> BacktestSources:
    """Fetch and compile what a backtest runs, before its JVM starts: classes join the JVM's classpath at start."""
    if strategy_commit and strategy_source:
        raise StrategySourceError("pass a strategy commit or a local strategy source, not both")
    engine = engine_jars(engine_version)
    sources = describe_engine(engine)
    engine_hash = sources.engine_jar_hash or _jar_hash(engine[0])

    if strategy_commit:
        sha = strategy_commit if FULL_SHA.match(strategy_commit) else resolve_strategy_commit(strategy_commit)
        classes, pinned = _strategy_commit_classes(sha, engine, engine_hash)
        sources.strategy_source = "commit"
        sources.strategy_commit = sha
        sources.strategy_dirty = False
    elif strategy_source:
        classes, head, pinned, dirty = _local_strategy_classes(strategy_source, engine)
        sources.strategy_source = "local"
        sources.strategy_commit = head
        sources.strategy_dirty = dirty
    else:
        return sources

    sources.classpath = [*engine, str(classes)]
    sources.platform_orchestrator = pinned
    if not pinned:
        sources.warnings.append("the strategy's strategies/pom.xml pins no platform.orchestrator; drift unchecked")
        return sources
    try:
        warning = drift_warning(sources, orchestrator_versions(pinned))
    except (StrategySourceError, requests.RequestException) as e:
        warning = f"couldn't check the engine against orchestrator {pinned}: {e}"
    if warning:
        sources.warnings.append(warning)
    return sources
