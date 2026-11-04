"""A session's Python strategy logs reach stdout, which the instance ships to CloudWatch."""
import subprocess
import sys

SCRIPT = """
import logging
from gnomepy.java.strategy.runner import _configure_logging
_configure_logging()
logging.getLogger("gnomepy_research.sessions.example").info("pricing with v1")
logging.getLogger("gnomepy_research.sessions.example").debug("not shown")
"""


def test_strategy_info_logs_go_to_stdout():
    out = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, check=True)
    assert "INFO gnomepy_research.sessions.example: pricing with v1" in out.stdout
    assert "not shown" not in out.stdout and out.stderr == ""
