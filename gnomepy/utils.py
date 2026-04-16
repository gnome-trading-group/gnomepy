from __future__ import annotations

import re
import secrets
import time
from datetime import datetime, timezone


def uuid7() -> str:
    """Generate a UUIDv7 (time-ordered, sortable)."""
    ms = int(time.time() * 1000) & 0xFFFFFFFFFFFF
    rand_a = secrets.randbits(12)
    rand_b = secrets.randbits(62)
    n = (ms << 80) | (0x7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b
    h = f"{n:032x}"
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _camel_to_kebab(name: str) -> str:
    """Convert CamelCase to kebab-case. E.g. 'MomentumTaker' → 'momentum-taker'."""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1-\2", name)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1-\2", s)
    return s.lower()


def _extract_class_name(strategy_name: str) -> str:
    """Extract the class name from a strategy identifier.

    Handles:
    - Python import path  "module.path:ClassName"  → "ClassName"
    - Java FQN            "com.example.MyStrategy" → "MyStrategy"
    - Plain name          "MomentumTaker"           → "MomentumTaker"
    """
    if ":" in strategy_name:
        return strategy_name.split(":", 1)[1]
    if "." in strategy_name:
        return strategy_name.rsplit(".", 1)[1]
    return strategy_name


def generate_backtest_id(strategy_name: str | None = None) -> str:
    """Generate a human-readable backtest ID.

    Format: ``{strategy_slug}-{YYYYMMDD}-{HHMMSS}-{4_hex_rand}``

    Examples::

        generate_backtest_id("MomentumTaker")
        # "momentum-taker-20260415-103000-a3b2"

        generate_backtest_id("gnomepy_research.strategies.momentum:MomentumTaker")
        # "momentum-taker-20260415-103000-a3b2"

        generate_backtest_id()
        # "backtest-20260415-103000-a3b2"
    """
    if strategy_name:
        slug = _camel_to_kebab(_extract_class_name(strategy_name))
    else:
        slug = "backtest"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    rand = secrets.token_hex(2)
    return f"{slug}-{ts}-{rand}"
