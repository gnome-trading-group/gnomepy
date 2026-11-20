"""Order role and episode columns in backtest results, read from a stand-in for the Java recorder — no JVM needed."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from gnomepy.java.recorder import BacktestResults


@dataclass(frozen=True)
class _Column:
    _name: str
    _type: str
    _index: int

    def name(self):
        return self._name

    def type(self):
        return self._type

    def columnIndex(self):
        return self._index


class _Buffer:
    def __init__(self, columns: dict[str, tuple[str, list]]):
        self._defs = [_Column(name, kind, i) for i, (name, (kind, _)) in enumerate(columns.items())]
        self._values = [values for _, values in columns.values()]

    def getCount(self):
        return len(self._values[0])

    def getColumns(self):
        return self._defs

    def getLongColumn(self, i):
        return self._values[i]

    getIntColumn = getDoubleColumn = getStringColumn = getLongColumn

    def getByteColumn(self, i):
        return bytes(self._values[i])


def _fills(with_roles: bool) -> _Buffer:
    columns = {
        "timestamp": ("LONG", [1, 2]), "exchange_id": ("INT", [4, 6]), "security_id": ("LONG", [10, 20]),
        "listing_id": ("INT", [100, 200]), "strategy_id": ("INT", [0, 0]), "client_oid": ("LONG", [7, 9]),
        "side": ("BYTE", [1, 1]), "fill_price": ("LONG", [400_000_000, 550_000_000]),
        "fill_qty": ("LONG", [10_000_000, 10_000_000]), "leaves_qty": ("LONG", [0, 0]), "fee": ("DOUBLE", [0.0, 0.1]),
        "book_bid_price": ("LONG", [400_000_000, 540_000_000]), "book_ask_price": ("LONG", [410_000_000, 550_000_000]),
        "liquidity": ("BYTE", [1, 2]),
    }
    if with_roles:
        columns["order_role"] = ("BYTE", [1, 2])
        columns["episode_id"] = ("LONG", [0, 7])
    return _Buffer(columns)


class _Recorder:
    def __init__(self, fills: _Buffer):
        self._fills = fills

    def getFillRecords(self):
        return self._fills

    def getFillRecordCount(self):
        return self._fills.getCount()


def test_fills_name_the_role_and_keep_the_episode():
    fills = BacktestResults(_Recorder(_fills(with_roles=True))).fills_df()

    assert fills["order_role"].tolist() == ["Quote", "Hedge"]
    assert fills["episode_id"].tolist() == [0, 7]
    hedge = fills[fills.order_role == "Hedge"].iloc[0]
    assert hedge.episode_id == fills[fills.order_role == "Quote"].iloc[0].client_oid


def test_results_from_older_engines_load_without_the_columns():
    fills = BacktestResults(_Recorder(_fills(with_roles=False))).fills_df()

    assert "order_role" not in fills.columns
    assert "episode_id" not in fills.columns
    assert fills["liquidity"].tolist() == ["Maker", "Taker"]


def test_roles_and_episodes_survive_save_and_load(tmp_path):
    fills = BacktestResults(_Recorder(_fills(with_roles=True))).fills_df()
    BacktestResults.from_dataframes(fills_df=fills).save(tmp_path)

    loaded = BacktestResults.from_parquet(tmp_path).fills_df()

    assert loaded["order_role"].tolist() == ["Quote", "Hedge"]
    assert loaded["episode_id"].tolist() == [0, 7]
    assert isinstance(loaded.index, pd.DatetimeIndex)
