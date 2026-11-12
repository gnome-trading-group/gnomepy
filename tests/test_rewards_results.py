"""Estimated venue rewards in backtest results and the report — no JVM needed."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy.java.recorder import BacktestResults
from gnomepy.metadata import BacktestMetadata
from gnomepy.reporting import plots
from gnomepy.reporting.report import BacktestReport

HOUR = 3600 * 1_000_000_000
T0 = pd.Timestamp("2026-11-11 05:00:00").value


def _snapshot(at: int, listing_id: int, program: str, start: int, end: int | None, gross: float, market_gross: float,
              market_paid: float, counting_hours: float) -> dict:
    return {
        "timestamp": pd.Timestamp(at), "listing_id": listing_id, "venue_program_id": program, "market": "KXRT-VIO-60",
        "period_start": pd.Timestamp(start), "period_end": pd.Timestamp(end) if end is not None else pd.NaT,
        "program_reward": 100.0, "gross": gross, "share_nanos": 0.0, "market_gross": market_gross,
        "market_paid": market_paid, "counting_nanos": int(counting_hours * HOUR), "one_sided_nanos": 0,
        "depth_unknown_nanos": 0, "no_data_nanos": 0,
    }


def _rewards() -> pd.DataFrame:
    rows = [
        # A day-long program that ended inside the run, earned on the YES (1) and NO (2) listings of one market.
        _snapshot(T0 + 12 * HOUR, 1, "k1", T0, T0 + 24 * HOUR, 3.0, 4.0, 0.0, 12),
        _snapshot(T0 + 12 * HOUR, 2, "k1", T0, T0 + 24 * HOUR, 1.0, 4.0, 0.0, 12),
        _snapshot(T0 + 30 * HOUR, 1, "k1", T0, T0 + 24 * HOUR, 6.009, 8.0, 8.0, 18),
        _snapshot(T0 + 30 * HOUR, 2, "k1", T0, T0 + 24 * HOUR, 1.991, 8.0, 8.0, 18),
        # A program still running when the run ended.
        _snapshot(T0 + 30 * HOUR, 1, "k2", T0 + 24 * HOUR, None, 0.5, 0.5, 0.0, 6),
    ]
    return pd.DataFrame(rows).set_index("timestamp")


def _fills(liquidity: list[str]) -> pd.DataFrame:
    ts = pd.to_datetime([T0 + i * HOUR for i in range(len(liquidity))])
    return pd.DataFrame({
        "listing_id": [9] * len(liquidity), "client_oid": range(len(liquidity)), "side": ["Bid"] * len(liquidity),
        "fill_price": [0.5] * len(liquidity), "fill_qty": [100.0] * len(liquidity), "liquidity": liquidity,
        "fee": [0.0] * len(liquidity),
    }, index=pd.DatetimeIndex(ts, name="timestamp"))


class _Registry:
    """The registry's GET /fees answer for listing 9: a Polymarket fee type with a 25% maker rebate."""

    def __init__(self, rebate: float | None):
        self.rebate = rebate
        self.known_as_of = None

    def get_fee_timelines(self, listing_ids, start=None, end=None, known_as_of=None):
        self.known_as_of = known_as_of
        definition = {"currency": "USD",
                      "components": [{"kind": "PRICE_CURVE", "taker": 0.04, "maker": 0, "exponent": 1}],
                      "rounding": {"mode": "NONE"}}
        if self.rebate is not None:
            definition["makerRebate"] = {"share": self.rebate}
        return {9: [{"effectiveFrom": "2026-01-01T00:00:00Z", "effectiveTo": None, "definition": definition}]}


def test_rewards_df_takes_each_programs_last_snapshot_and_pays_only_periods_that_ended():
    df = BacktestResults.from_dataframes(rewards_df=_rewards()).rewards_df().set_index(["listing_id", "venue_program_id"])

    assert df.loc[(1, "k1"), "gross"] == 6.009
    assert df.loc[(1, "k1"), "closed"] and df.loc[(2, "k1"), "closed"]
    # The market's $8.00 split by gross, 6.009 : 1.991.
    assert np.isclose(df.loc[(1, "k1"), "paid"], 8.0 * 6.009 / 8.0)
    assert np.isclose(df.loc[(2, "k1"), "paid"], 8.0 * 1.991 / 8.0)
    assert not df.loc[(1, "k2"), "closed"]
    assert np.isnan(df.loc[(1, "k2"), "paid"])
    assert np.isclose(df.loc[(1, "k1"), "counting_fraction"], 18 / 24)
    # An open period runs to the end of the run: 6 h of the 6 h since it started.
    assert np.isclose(df.loc[(1, "k2"), "counting_fraction"], 1.0)


def test_a_market_under_the_venues_minimum_pays_nothing_on_any_listing():
    rows = [_snapshot(T0 + 30 * HOUR, 1, "k1", T0, T0 + 24 * HOUR, 0.6, 0.9, 0.0, 18),
            _snapshot(T0 + 30 * HOUR, 2, "k1", T0, T0 + 24 * HOUR, 0.3, 0.9, 0.0, 18)]
    df = BacktestResults.from_dataframes(rewards_df=pd.DataFrame(rows).set_index("timestamp")).rewards_df()

    assert (df["paid"] == 0).all()


def test_maker_fills_earn_the_rebate_share_of_the_taker_fee_they_would_have_paid():
    metadata = BacktestMetadata(backtest_id="t", simulation_as_of="2026-11-12T00:00:00Z")
    results = BacktestResults.from_dataframes(fills_df=_fills(["Maker", "Taker", "Maker"]), metadata=metadata)
    registry = _Registry(rebate=0.25)

    rebates = results.maker_rebates_df(registry=registry)

    assert registry.known_as_of == "2026-11-12T00:00:00Z"
    assert len(rebates) == 2
    # Taker fee on 100 at $0.50: 0.04 × 100 × 0.25 = $1.00; a quarter of it.
    assert np.allclose(rebates["rebate"], 0.25)


def test_listings_without_a_maker_rebate_earn_none():
    results = BacktestResults.from_dataframes(fills_df=_fills(["Maker"]))

    assert results.maker_rebates_df(registry=_Registry(rebate=None)).empty


def test_the_report_lists_the_rewards_and_sums_them_in_the_summary():
    results = BacktestResults.from_dataframes(rewards_df=_rewards())
    report = BacktestReport(results)

    summary = report.summary()
    assert np.isclose(summary["rewards_estimated_gross"], 6.009 + 1.991 + 0.5)
    assert np.isclose(summary["rewards_estimated_paid"], 8.0)
    assert summary["maker_rebates_estimated"] == 0.0
    html = plots._render_rewards(report)
    assert "Liquidity rewards: $8.50 gross, $8.00 paid" in html
    assert "open" in html


def test_a_run_without_rewards_has_no_section():
    assert plots._render_rewards(BacktestReport(BacktestResults.from_dataframes())) is None


def _market() -> pd.DataFrame:
    ts = pd.to_datetime([T0 + h * HOUR for h in (1, 13, 29)])
    return pd.DataFrame({"listing_id": [1] * 3, "exchange_id": [5] * 3, "security_id": [7] * 3,
                         "bid_price_0": [0.29] * 3, "ask_price_0": [0.31] * 3, "bid_size_0": [10.0] * 3,
                         "ask_size_0": [10.0] * 3}, index=pd.DatetimeIndex(ts, name="timestamp"))


def test_the_report_draws_rewards_and_pnl_with_rewards_and_shows_them_in_the_cards():
    report = BacktestReport(BacktestResults.from_dataframes(market_df=_market(), rewards_df=_rewards()))

    curve = report.rewards_curve
    # Each snapshot sums every listing and program: 3.0 + 1.0 at 12 h, then 6.009 + 1.991 + 0.5 at 30 h.
    assert np.isclose(curve.loc[pd.Timestamp(T0 + 12 * HOUR)], 4.0)
    assert np.isclose(curve.iloc[-1], 8.5)
    with_rewards = report.pnl_with_rewards_curve
    assert np.isclose(with_rewards.iloc[-1], report.pnl_curve.iloc[-1] + 8.5)

    summary = report.summary()
    assert np.isclose(summary["rewards_estimated_total"], 8.5)
    assert np.isclose(summary["final_pnl_with_rewards_estimated"], summary["final_pnl"] + 8.5)
    names = [t.name for t in plots.plot_pnl(report, show_mid=False).data]
    assert "Rewards (est.)" in names and "PnL + rewards (est.)" in names
    cards = plots._kpi_cards_html(summary)
    assert "Rewards (est.)" in cards and "PnL + rewards (est.)" in cards


def test_a_run_without_rewards_draws_and_shows_none_of_it():
    report = BacktestReport(BacktestResults.from_dataframes(market_df=_market()))

    assert report.rewards_curve.empty
    names = [t.name for t in plots.plot_pnl(report, show_mid=False).data]
    assert "Rewards (est.)" not in names
    assert "Rewards (est.)" not in plots._kpi_cards_html(report.summary())
