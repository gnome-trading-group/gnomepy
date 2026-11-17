"""A report values positions still held at the end at their settlement price once the registry has one."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from gnomepy.java.recorder import BacktestResults
from gnomepy.reporting import plots
from gnomepy.reporting.report import BacktestReport

MINUTE = 60 * 1_000_000_000
T0 = pd.Timestamp("2026-11-13 16:00:00").value


class _Registry:
    """Listings 1..n on securities 11..; ``settle`` maps a security to its scaled settlement price (None: unsettled)."""

    def __init__(self, settle: dict[int, int | None], fail: bool = False):
        self.settle = settle
        self.fail = fail
        self.calls = 0

    def get_listing(self, listing_id=None, **_):
        self.calls += 1
        if self.fail:
            raise ConnectionError("registry unreachable")
        return [SimpleNamespace(listing_id=listing_id, security_id=listing_id + 10)]

    def get_event_contracts(self, security_id=None, **_):
        self.calls += 1
        if security_id not in self.settle:
            return []
        price = self.settle[security_id]
        # The registry sends bigint columns as strings.
        return [SimpleNamespace(security_id=security_id, settlement_price=None if price is None else str(price),
                                settled_at=None if price is None else "2026-11-13T19:02:11Z")]


def _market(mids: dict[int, list[float]]) -> pd.DataFrame:
    rows = []
    for listing_id, values in mids.items():
        for i, mid in enumerate(values):
            rows.append({"timestamp": pd.Timestamp(T0 + i * MINUTE), "listing_id": listing_id, "exchange_id": 4,
                         "security_id": listing_id + 10, "bid_price_0": mid - 0.01, "ask_price_0": mid + 0.01,
                         "bid_size_0": 100.0, "ask_size_0": 100.0})
    return pd.DataFrame(rows).set_index("timestamp").sort_index(kind="stable")


def _fills(rows: list[tuple[int, str, float, float, float]]) -> pd.DataFrame:
    """(listing, side, price, qty, fee), one a minute from the start."""
    return pd.DataFrame({
        "listing_id": [r[0] for r in rows], "client_oid": range(len(rows)), "side": [r[1] for r in rows],
        "fill_price": [r[2] for r in rows], "fill_qty": [r[3] for r in rows], "liquidity": ["Taker"] * len(rows),
        "fee": [r[4] for r in rows],
    }, index=pd.DatetimeIndex([pd.Timestamp(T0 + i * MINUTE) for i in range(len(rows))], name="timestamp"))


def _locked_pair_report(registry, **kwargs) -> BacktestReport:
    # 100 of team A at 0.48 on listing 1 and 100 of team B at 0.50 on listing 2, $1 fee each; the books end at mids
    # 0.95 and 0.03, so the pair marks at 0.98 though it pays $1.
    market = _market({1: [0.48, 0.70, 0.95], 2: [0.50, 0.28, 0.03]})
    fills = _fills([(1, "Bid", 0.48, 100.0, 1.0), (2, "Bid", 0.50, 100.0, 1.0)])
    return BacktestReport(BacktestResults.from_dataframes(market_df=market, fills_df=fills), registry=registry,
                          **kwargs)


def test_a_locked_pair_held_to_settlement_is_worth_what_it_paid_not_its_last_marks():
    report = _locked_pair_report(_Registry({11: 1_000_000_000, 12: 0}))

    summary = report.summary()

    # Paid 48 + 50 + 2 in fees for a pair that pays 100.
    assert np.isclose(summary["final_pnl"], 0.0)
    assert np.isclose(summary["final_pnl_marked"], (95 - 48 - 1) + (3 - 50 - 1))
    assert summary["settlement"] == {"status": "settled", "settled": {"1": 1.0, "2": 0.0}, "unsettled": []}
    assert np.isclose(summary["final_pnl_by_symbol"]["1"], 100 - 48 - 1)
    assert np.isclose(summary["final_pnl_by_symbol"]["2"], -50 - 1)


def test_each_listing_takes_its_own_venues_settlement():
    # Kalshi-style partial settlement: 0.46 and 0.54 rather than 1 and 0.
    report = _locked_pair_report(_Registry({11: 460_000_000, 12: 540_000_000}))

    assert np.isclose(report.summary()["final_pnl"], 46 + 54 - 48 - 50 - 2)


def test_a_listing_not_settled_yet_keeps_its_last_mark():
    report = _locked_pair_report(_Registry({11: 1_000_000_000, 12: None}))

    summary = report.summary()

    assert np.isclose(summary["final_pnl"], (100 - 48 - 1) + (3 - 50 - 1))
    assert summary["settlement"] == {"status": "partial", "settled": {"1": 1.0}, "unsettled": [2]}


def test_without_the_registry_the_report_still_renders_at_the_last_marks():
    report = _locked_pair_report(_Registry({}, fail=True))

    summary = report.summary()

    assert summary["settlement"]["status"] == "unavailable"
    assert np.isclose(summary["final_pnl"], summary["final_pnl_marked"])
    assert report.to_html()


def test_a_flat_run_asks_the_registry_nothing():
    market = _market({1: [0.48, 0.52]})
    fills = _fills([(1, "Bid", 0.48, 10.0, 0.0), (1, "Ask", 0.52, 10.0, 0.0)])
    registry = _Registry({11: 1_000_000_000})

    summary = BacktestReport(BacktestResults.from_dataframes(market_df=market, fills_df=fills),
                             registry=registry).summary()

    assert registry.calls == 0
    assert summary["settlement"]["status"] == "flat"
    assert np.isclose(summary["final_pnl"], 0.4)


def test_a_windowed_report_stays_at_the_marks():
    report = _locked_pair_report(_Registry({11: 1_000_000_000, 12: 0}),
                                 start_date=pd.Timestamp(T0), end_date=pd.Timestamp(T0 + MINUTE))

    summary = report.summary()

    assert summary["settlement"]["status"] == "windowed"
    assert np.isclose(summary["final_pnl"], summary["final_pnl_marked"])


def test_the_report_shows_the_settled_pnl_and_the_last_mark_beside_it():
    report = _locked_pair_report(_Registry({11: 1_000_000_000, 12: 0}))

    cards = plots._kpi_cards_html(report.summary())
    trace = next(t for t in plots.plot_pnl(report, show_mid=False).data if t.name == "At settlement")

    assert "Final PnL" in cards and "PnL at last mark" in cards
    assert np.isclose(trace.y[0], 0.0)
