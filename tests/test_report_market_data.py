"""Missing market data in the backtest report: shaded on the time charts, listed in its own section — no JVM needed."""
from __future__ import annotations

import pandas as pd

from gnomepy.java.recorder import BacktestResults
from gnomepy.metadata import BacktestMetadata
from gnomepy.reporting import plots
from gnomepy.reporting.report import BacktestReport

T0 = pd.Timestamp("2026-10-28 12:00:00").value
MIN = 60 * 1_000_000_000


def _market_df() -> pd.DataFrame:
    ts = pd.to_datetime([T0 + 5 * MIN, T0 + 6 * MIN])
    return pd.DataFrame(
        {"listing_id": [7, 7], "exchange_id": [1, 1], "security_id": [101, 101], "bid_price_0": [0.4, 0.41], "ask_price_0": [0.5, 0.5],
         "bid_size_0": [1.0, 1.0], "ask_size_0": [1.0, 1.0]},
        index=pd.DatetimeIndex(ts, name="timestamp"),
    )


def _market_data(streams: list[dict], untracked_ns: int = 0) -> dict:
    missing_ns = sum(s["missing_ns"] for s in streams)
    return {"on_missing": "mark", "missing_ns": missing_ns, "untracked_ns": untracked_ns, "missing_fraction": 0.25,
            "marks_sent": 1, "marks_without_book": 1, "streams": streams}


def _stream(listing_id: int, stretches: list[tuple[int, int, str]]) -> dict:
    return {"stream_id": listing_id + 1, "listing_id": listing_id, "untracked_ns": 0,
            "missing_ns": sum(to - frm for frm, to, _ in stretches),
            "missing": [{"from_ns": frm, "to_ns": to, "reason": reason} for frm, to, reason in stretches]}


def _report(market_data: dict | None) -> BacktestReport:
    metadata = BacktestMetadata(backtest_id="test", market_data=market_data)
    return BacktestReport(BacktestResults.from_dataframes(market_df=_market_df(), metadata=metadata))


ONE_LISTING = _market_data([_stream(7, [(T0, T0 + 4 * MIN, "NOT_COLLECTED"), (T0 + 10 * MIN, T0 + 12 * MIN, "TASK_LOST")])])


def test_each_stretch_is_shaded_on_every_row_with_a_marker_saying_why():
    fig = plots.plot_pnl(_report(ONE_LISTING), show_mid=False)

    bands = [s for s in fig.layout.shapes if s.type == "rect"]
    assert len(bands) == 2 * 2, "two stretches across PnL and drawdown"
    markers = [a for a in fig.layout.annotations if a.hovertext]
    assert len(markers) == 2
    assert "task lost" in markers[1].hovertext and "12:10:00 for 2.0 min" in markers[1].hovertext
    assert "market data missing" in [t.name for t in fig.data]


def test_without_market_data_nothing_is_shaded():
    fig = plots.plot_spread(_report(None))

    assert not fig.layout.shapes
    assert "market data missing" not in [t.name for t in fig.data]


def test_several_listings_each_get_a_legend_entry():
    two = _market_data([_stream(7, [(T0, T0 + MIN, "STALE")]), _stream(9, [(T0, T0 + MIN, "STALE")])])

    fig = plots.plot_spread(_report(two))

    assert [t.name for t in fig.data if t.name.startswith("missing:")] == ["missing: 7", "missing: 9"]
    assert sorted(a.yshift for a in fig.layout.annotations if a.hovertext) == [0, 10]


def test_too_many_stretches_join_near_neighbours():
    many = [plots._MissingStretch(7, pd.Timestamp(T0 + i * 1000), pd.Timestamp(T0 + i * 1000 + 500), "STALE")
            for i in range(plots._MAX_MISSING_BANDS + 1)]

    bands = plots._bands(many)

    assert len(bands) == 1
    assert (bands[0].start, bands[0].end) == (many[0].start, many[-1].end)


def test_the_market_data_section_lists_each_stretch_and_the_totals():
    html = plots._render_market_data(_report({**ONE_LISTING, "untracked_ns": 30 * MIN}))

    assert "Missing 6.0 min (25.0% of the run" in html
    assert "30.0 min predates collection coverage" in html
    assert html.count("<tr><td>") == 2
    assert "2026-10-28 12:10:00" in html and "task lost" in html
    assert plots._render_market_data(_report(None)) is None


def test_the_summary_table_formats_the_missing_fraction_and_shows_none_as_a_dash():
    assert "25.0%" in plots._summary_table_html({"market_data_missing_fraction": 0.25})
    assert "<td>—</td>" in plots._summary_table_html({"market_data_missing_fraction": None})
