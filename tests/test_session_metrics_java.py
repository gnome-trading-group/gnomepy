"""A live or paper session's strategy metrics, journaled and read back. Needs the gnome-backtest uber JAR, whose
classpath holds the same strategy and journal classes the orchestrator runs; skipped without it."""
from __future__ import annotations

import jpype
import pytest

from gnomepy.java._classpath import discover_classpath
from gnomepy.java._jvm import ensure_jvm_started
from gnomepy.java.backtest.strategy import Strategy
from gnomepy.java.strategy.runner import _create_strategy_callback
from gnomepy.journal import read_journal_metrics


def _jar_available() -> bool:
    try:
        return len(discover_classpath("gnome-backtest")) > 0
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not _jar_available(), reason="gnome-backtest JAR not available")

# JournalSource ids the orchestrator gives these buffers.
_INTENT, _EXEC_REPORT, _MARKET_DATA, _METRIC = 1, 2, 7, 8


@pytest.fixture(scope="module", autouse=True)
def _jvm():
    ensure_jvm_started()


class _DecisionStrategy(Strategy):
    """Records the book it saw on every update, the way CS2Series records each decision."""

    def __init__(self):
        self.updates = 0

    def register_metrics(self):
        stream = self.metrics.create_stream("decisions")
        self.ts_col = stream.add_long_column("timestamp")
        self.bid_col = stream.add_double_column("best_bid")
        self.level_col = stream.add_int_column("update")
        stream.freeze()
        self.decisions = stream
        heartbeat = self.metrics.create_stream("heartbeat")
        self.beat_col = heartbeat.add_long_column("updates")
        heartbeat.freeze()
        self.heartbeat = heartbeat

    def on_market_data(self, data):
        self.updates += 1
        self.decisions.begin()
        self.decisions.set_long(self.ts_col, data.event_timestamp)
        self.decisions.set_double(self.bid_col, data.bid_price(0) / 1e9)
        self.decisions.set_int(self.level_col, self.updates)
        self.decisions.commit()
        if self.updates % 2 == 0:
            self.heartbeat.begin()
            self.heartbeat.set_long(self.beat_col, self.updates)
            self.heartbeat.commit()
        return []

    def on_execution_report(self, report):
        return []


class _Session:
    """The strategy's side of a session as the orchestrator wires it: its buffers on one sequence, all journaled."""

    def __init__(self, journal_dir, strategy: Strategy):
        sequence = jpype.JClass("group.gnometrading.sequencer.GlobalSequence")()
        ring = jpype.JClass("group.gnometrading.sequencer.SequencedRingBuffer")
        self.market_data = ring(jpype.JClass("group.gnometrading.schemas.Mbp10Schema"), sequence)
        exec_reports = ring(jpype.JClass("group.gnometrading.schemas.OrderExecutionReport"), sequence)
        intents = ring(jpype.JClass("group.gnometrading.schemas.Intent"), sequence)
        metrics = ring(jpype.JClass("group.gnometrading.schemas.MetricRow"), sequence)

        self.segments = []
        self.writer = jpype.JClass("group.gnometrading.sequencer.JournalWriter")(
            jpype.JClass("java.nio.file.Path").of(str(journal_dir)), 1 << 20, self.segments.append
        )
        self.pollers = [
            buffer.createPoller(self.writer.source(source))
            for buffer, source in (
                (self.market_data, _MARKET_DATA),
                (exec_reports, _EXEC_REPORT),
                (intents, _INTENT),
                (metrics, _METRIC),
            )
        ]
        PythonStrategyAgent = jpype.JClass("group.gnometrading.strategies.PythonStrategyAgent")
        self.agent = PythonStrategyAgent.createWithBuffers(
            0, self.market_data, exec_reports, intents, None, None, _create_strategy_callback(strategy)
        )
        recorder = jpype.JClass("group.gnometrading.sequencer.metrics.JournaledMetricRecorder")(metrics)
        self.agent.configureMetrics(recorder)

    def tick(self, timestamp: int, bid: float):
        schema = self.market_data.claim()
        schema.encoder.timestampEvent(timestamp).bidPrice0(int(bid * 1e9))
        self.market_data.publish()
        self.agent.doWork()
        self.journal()

    def journal(self):
        while sum(int(poller.poll()) for poller in self.pollers) > 0:
            pass

    def close(self):
        self.journal()
        self.writer.close()
        for segment in self.segments:
            segment.seal()


def test_rows_recorded_in_on_market_data_come_back_from_the_journal(tmp_path):
    strategy = _DecisionStrategy()
    session = _Session(tmp_path, strategy)

    session.tick(1_000, 0.41)
    session.tick(2_000, 0.42)
    session.tick(3_000, 0.40)
    session.close()

    metrics = read_journal_metrics(tmp_path)

    decisions = metrics["decisions"]
    assert list(decisions.columns) == ["timestamp", "best_bid", "update"]
    assert decisions["timestamp"].tolist() == [1_000, 2_000, 3_000]
    assert decisions["best_bid"].tolist() == pytest.approx([0.41, 0.42, 0.40])
    assert decisions["update"].tolist() == [1, 2, 3]
    assert str(decisions["update"].dtype) == "int32"
    assert metrics["heartbeat"]["updates"].tolist() == [2]
    # Each row sits right after the update it was recorded for: the declarations, then update, row, update, row.
    assert decisions.index.name == "sequence"
    assert decisions.index.tolist() == [8, 10, 13]
    assert metrics["heartbeat"].index.tolist() == [11]


def test_rows_in_the_last_segment_are_kept_when_the_session_exits(tmp_path):
    strategy = _DecisionStrategy()
    session = _Session(tmp_path, strategy)

    session.tick(1_000, 0.41)
    session.writer.roll()
    session.tick(2_000, 0.42)
    session.close()

    assert len(session.segments) == 2
    assert read_journal_metrics(tmp_path)["decisions"]["timestamp"].tolist() == [1_000, 2_000]


def test_a_stream_declared_but_never_recorded_is_empty(tmp_path):
    strategy = _DecisionStrategy()
    session = _Session(tmp_path, strategy)

    session.tick(1_000, 0.41)
    session.close()

    heartbeat = read_journal_metrics(tmp_path)["heartbeat"]
    assert heartbeat.empty
    assert list(heartbeat.columns) == ["updates"]
