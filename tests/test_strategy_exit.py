from datetime import timedelta

from gnomepy.java.backtest.strategy import Strategy


class _Quiet(Strategy):
    def on_market_data(self, data):
        return []

    def on_execution_report(self, report):
        return []


class _FakeControl:
    """Stands in for the Java StrategyControl the session hands over in onInit."""

    def __init__(self):
        self.requests = []

    def requestExit(self, reason, flatten, timeout_millis):
        self.requests.append((reason, flatten, timeout_millis))
        return len(self.requests) == 1

    def isExiting(self):
        return bool(self.requests)


def test_request_exit_passes_the_reason_flatten_and_timeout_to_the_session():
    strategy = _Quiet()
    strategy._control = _FakeControl()

    assert strategy.request_exit("match went live", flatten=True, timeout=timedelta(seconds=90))

    assert strategy._control.requests == [("match went live", True, 90_000)]
    assert strategy.is_exiting


def test_request_exit_keeps_positions_by_default():
    strategy = _Quiet()
    strategy._control = _FakeControl()

    strategy.request_exit("done")

    assert strategy._control.requests == [("done", False, 300_000)]


def test_only_the_first_request_counts():
    strategy = _Quiet()
    strategy._control = _FakeControl()

    assert strategy.request_exit("first")
    assert not strategy.request_exit("second")


def test_request_exit_before_init_is_ignored():
    strategy = _Quiet()

    assert not strategy.request_exit("too early")
    assert not strategy.is_exiting
