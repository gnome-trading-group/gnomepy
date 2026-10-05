"""Windowed reports, mid price, Sharpe/Sortino, drawdown and result-frame scaling — no JVM needed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gnomepy.java.recorder import BacktestResults
from gnomepy.java.statics import Scales
from gnomepy.reporting.metrics import compute_max_drawdown, compute_sharpe, mid_price
from gnomepy.reporting.report import BacktestReport

PRICE_SCALE = 1_000_000_000
SIZE_SCALE = 1_000_000


@pytest.fixture(autouse=True)
def _scales(monkeypatch):
    monkeypatch.setattr(Scales, "PRICE", PRICE_SCALE)
    monkeypatch.setattr(Scales, "SIZE", SIZE_SCALE)


def _ts(*seconds: int) -> pd.DatetimeIndex:
    base = pd.Timestamp("2026-01-23 10:00:00", tz="UTC")
    return pd.DatetimeIndex([base + pd.Timedelta(seconds=s) for s in seconds], name="timestamp")


def _market(mids: list[float], seconds: list[int], security_id: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "exchange_id": [1] * len(mids),
            "security_id": [security_id] * len(mids),
            "bid_price_0": [m - 0.5 for m in mids],
            "ask_price_0": [m + 0.5 for m in mids],
        },
        index=_ts(*seconds),
    )


def _fill(second: int, side: str, qty: float, price: float, fee: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {"exchange_id": [1], "security_id": [1], "side": [side], "fill_qty": [qty], "fill_price": [price], "fee": [fee]},
        index=_ts(second),
    )


class TestWindowedReport:
    def test_inventory_bought_before_the_window_is_marked_inside_it(self):
        market = _market([100.0, 100.0, 110.0, 120.0], [0, 10, 20, 30])
        fills = _fill(0, "Bid", 1.0, 100.0)
        results = BacktestResults.from_dataframes(market_df=market, fills_df=fills)

        report = BacktestReport(results, start_date=_ts(10)[0], end_date=_ts(30)[0])

        assert report.position_curve.iloc[0] == 1.0
        # Window PnL is the move from 100 to 120 on the one unit held.
        assert report.pnl_curve.iloc[-1] == pytest.approx(20.0)
        assert report.summary()["fill_count"] == 0

    def test_sell_inside_the_window_closes_the_carried_position(self):
        market = _market([100.0, 100.0, 110.0], [0, 10, 20])
        fills = pd.concat([_fill(0, "Bid", 1.0, 100.0), _fill(20, "Ask", 1.0, 110.0)])
        results = BacktestResults.from_dataframes(market_df=market, fills_df=fills)

        report = BacktestReport(results, start_date=_ts(10)[0], end_date=_ts(20)[0])

        assert report.position_curve.iloc[-1] == 0.0
        assert report.pnl_curve.iloc[-1] == pytest.approx(10.0)
        assert report.summary()["fill_count"] == 1


class TestMidPrice:
    def test_one_sided_book_keeps_the_last_two_sided_mid(self):
        df = _market([100.0, 101.0], [0, 1])
        df.loc[df.index[1], "ask_price_0"] = np.nan
        assert list(mid_price(df)) == [100.0, 100.0]

    def test_mid_does_not_leak_across_listings(self):
        a = _market([100.0], [0], security_id=1)
        b = _market([50.0], [1], security_id=2)
        b.loc[b.index[0], ["bid_price_0", "ask_price_0"]] = np.nan
        mids = mid_price(pd.concat([a, b]))
        assert mids.iloc[0] == 100.0
        assert np.isnan(mids.iloc[1])


class TestSharpe:
    def test_empty_bars_count_as_flat_bars(self):
        # Three ticks over 60s: dropping empty 10s bars would leave 2 returns; carrying PnL forward gives 6.
        pnl = pd.Series([0.0, 1.0, 2.0], index=_ts(0, 30, 60))
        assert compute_sharpe(pnl, bar="10s")["n_bars"] == 6

    def test_sortino_uses_downside_deviation_over_all_bars(self):
        returns = np.array([1.0, -1.0, 2.0, -2.0])
        pnl = pd.Series(np.concatenate([[0.0], np.cumsum(returns)]), index=_ts(0, 10, 20, 30, 40))
        metrics = compute_sharpe(pnl, bar="10s")
        downside = np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2))
        assert metrics["sortino"] == pytest.approx(returns.mean() / downside)

    def test_leftover_bars_land_in_the_last_bucket(self):
        pnl = pd.Series(np.arange(12, dtype=float) ** 1.5, index=_ts(*range(0, 120, 10)))
        assert len(compute_sharpe(pnl, bar="10s", n_buckets=5)["sharpe_per_bucket"]) == 5


class TestMaxDrawdown:
    def test_largest_peak_to_trough(self):
        pnl = pd.Series([0.0, 5.0, 2.0, 8.0, 1.0, 4.0], index=_ts(*range(6)))
        assert compute_max_drawdown(pnl) == 7.0

    def test_reported_in_summary(self):
        results = BacktestResults.from_dataframes(
            market_df=_market([100.0, 90.0], [0, 10]), fills_df=_fill(0, "Bid", 1.0, 100.0)
        )
        assert BacktestReport(results).summary()["max_drawdown"] == pytest.approx(10.0)


class TestResultScaling:
    def _orders(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"submit_price": [1.25], "avg_fill_price": [1.25], "submit_size": [2.0], "filled_qty": [2.0],
             "leaves_qty": [0.0], "side": ["Bid"]},
            index=_ts(0),
        )

    def test_raw_request_does_not_change_the_default(self):
        results = BacktestResults.from_dataframes(orders_df=self._orders())
        raw = results.orders_df(scale_prices=False)
        assert raw["submit_price"].iloc[0] == 1_250_000_000
        assert raw["filled_qty"].iloc[0] == 2_000_000
        assert results.orders_df()["submit_price"].iloc[0] == 1.25

    def test_null_book_sides_read_as_nan(self):
        market = pd.DataFrame(
            {"exchange_id": [1], "security_id": [1], "bid_price_0": [np.nan], "ask_price_0": [0.5]}, index=_ts(0)
        )
        raw = BacktestResults.from_dataframes(market_df=market).market_records_df(scale_prices=False)
        assert np.isnan(raw["bid_price_0"].iloc[0])
        assert raw["ask_price_0"].iloc[0] == 500_000_000


class _Column:
    def __init__(self, name, idx):
        self._name, self._idx = name, idx

    def name(self):
        return self._name

    def type(self):
        return "LONG"

    def columnIndex(self):
        return self._idx


class _FakeMarketRecorder:
    """Stands in for the Java recorder: one market buffer of LONG columns."""

    def __init__(self, columns: dict[str, list[int]]):
        self._columns = [_Column(name, i) for i, name in enumerate(columns)]
        self._values = list(columns.values())

    def getMarketRecordCount(self):
        return len(self._values[0])

    def getMarketRecords(self):
        return self

    def getCount(self):
        return len(self._values[0])

    def getColumns(self):
        return self._columns

    def getLongColumn(self, idx):
        return self._values[idx]


def test_sbe_nulls_from_the_java_recorder_become_nan():
    null = np.iinfo(np.int64).min
    recorder = _FakeMarketRecorder({
        "timestamp": [1, 2],
        "exchange_id": [1, 1],
        "security_id": [1, 1],
        "bid_price_0": [null, 400_000_000],
        "ask_price_0": [500_000_000, 600_000_000],
        "bid_size_0": [null, 3_000_000],
        "ask_size_0": [1_000_000, 1_000_000],
        "last_trade_price": [null, null],
        "last_trade_size": [null, null],
    })
    df = BacktestResults(recorder).market_records_df()
    assert np.isnan(df["bid_price_0"].iloc[0])
    assert np.isnan(df["bid_size_0"].iloc[0])
    assert df["bid_price_0"].iloc[1] == pytest.approx(0.4)
    assert df["last_trade_price"].isna().all()


class _FakeOrderRecorder:
    """One order buffer of LONG/BYTE columns standing in for the Java recorder."""

    def __init__(self, columns: dict[str, tuple[str, list]]):
        self._columns = [_TypedColumn(name, t, i) for i, (name, (t, _)) in enumerate(columns.items())]
        self._values = [vals for _, vals in columns.values()]

    def getOrderRecordCount(self):
        return len(self._values[0])

    def getOrderRecords(self):
        return self

    def getCount(self):
        return len(self._values[0])

    def getColumns(self):
        return self._columns

    def getLongColumn(self, idx):
        return self._values[idx]

    def getByteColumn(self, idx):
        return self._values[idx]


class _TypedColumn(_Column):
    def __init__(self, name, col_type, idx):
        super().__init__(name, idx)
        self._type = col_type

    def type(self):
        return self._type


def test_market_order_price_is_nan_not_the_sbe_null():
    null = np.iinfo(np.int64).min
    recorder = _FakeOrderRecorder({
        "submit_timestamp": ("LONG", [1, 2]),
        "ack_timestamp": ("LONG", [0, 0]),
        "terminal_timestamp": ("LONG", [3, 4]),
        "side": ("BYTE", [1, 2]),
        "order_type": ("BYTE", [0, 1]),
        "final_status": ("BYTE", [0, 0]),
        "submit_price": ("LONG", [500_000_000, null]),
        "final_price": ("LONG", [510_000_000, null]),
        "submit_size": ("LONG", [1_000_000, 2_000_000]),
        "final_size": ("LONG", [1_000_000, 2_000_000]),
        "filled_qty": ("LONG", [1_000_000, 2_000_000]),
        "leaves_qty": ("LONG", [0, 0]),
        "total_cost": ("LONG", [500_000_000, 1_000_000_000]),
    })
    df = BacktestResults(recorder).orders_df()
    assert df["submit_price"].iloc[0] == pytest.approx(0.5)
    assert df["final_price"].iloc[0] == pytest.approx(0.51)
    assert np.isnan(df["submit_price"].iloc[1])
    assert np.isnan(df["final_price"].iloc[1])
