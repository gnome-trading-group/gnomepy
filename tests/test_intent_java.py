"""Intent construction — requires the gnome-backtest uber JAR; skipped without it."""
from __future__ import annotations

import pytest

from gnomepy.java._classpath import discover_classpath
from gnomepy.java._jvm import ensure_jvm_started
from gnomepy.java.enums import OrderType, Side
from gnomepy.java.oms import Intent


def _jar_available() -> bool:
    try:
        return len(discover_classpath("gnome-backtest")) > 0
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not _jar_available(), reason="gnome-backtest JAR not available")


@pytest.fixture(scope="module", autouse=True)
def _jvm():
    ensure_jvm_started()


def test_take_without_an_order_type_is_a_clear_error():
    with pytest.raises(ValueError, match="take_order_type"):
        Intent(1, 100, take_side=Side.BID, take_size=5)


def test_take_without_a_side_is_a_clear_error():
    with pytest.raises(ValueError, match="take_side"):
        Intent(1, 100, take_size=5, take_order_type=OrderType.MARKET)


def test_complete_take_and_no_take_still_build():
    Intent(1, 100, take_side=Side.ASK, take_size=5, take_order_type=OrderType.MARKET)
    Intent(1, 100, bid_price=10, bid_size=1)
