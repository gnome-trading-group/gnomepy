import zipfile

import pytest

from gnomepy.java._jvm import session_jvm_options


def _jar(path, entries: dict[str, str]) -> str:
    with zipfile.ZipFile(path, "w") as jar:
        for name, text in entries.items():
            jar.writestr(name, text)
    return str(path)


def test_reads_the_orchestrator_jar_skipping_comments_and_blanks(tmp_path):
    other = _jar(tmp_path / "other.jar", {"META-INF/MANIFEST.MF": "Manifest-Version: 1.0\n"})
    orchestrator = _jar(tmp_path / "gnome-orchestrator.jar", {
        "session-jvm-options": "# why\n-XX:+UseZGC\n\n  -XX:ConcGCThreads=2  \n",
    })

    assert session_jvm_options([str(tmp_path), other, orchestrator]) == ["-XX:+UseZGC", "-XX:ConcGCThreads=2"]


def test_refuses_a_classpath_without_them(tmp_path):
    old = _jar(tmp_path / "gnome-orchestrator-1.39.0.jar", {"META-INF/MANIFEST.MF": "Manifest-Version: 1.0\n"})

    with pytest.raises(RuntimeError, match="session-jvm-options"):
        session_jvm_options([old])
