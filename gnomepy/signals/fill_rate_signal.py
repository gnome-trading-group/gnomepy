from __future__ import annotations

from collections import deque


class FillRateSignal:
    """Tracks fill rate over a rolling window and provides a spread adjustment factor.

    High fill rate (above target) → adjustment > 1 → widen spreads.
    Low fill rate (below target) → adjustment < 1 → tighten spreads.

    Args:
        target_fill_rate: Desired fills per minute.
        window_sec: Rolling window size in seconds.
        min_scale: Minimum spread adjustment factor.
        max_scale: Maximum spread adjustment factor.
    """

    def __init__(
        self,
        target_fill_rate: float = 20.0,
        window_sec: float = 60.0,
        min_scale: float = 0.5,
        max_scale: float = 3.0,
    ):
        self.target_fill_rate = target_fill_rate
        self.window_ns = int(window_sec * 1_000_000_000)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self._fill_times: deque[int] = deque()

    def on_fill(self, timestamp_ns: int) -> None:
        """Record a fill timestamp (nanoseconds)."""
        self._fill_times.append(timestamp_ns)
        self._evict(timestamp_ns)

    def get_fill_rate(self, current_ns: int) -> float:
        """Returns fills per minute over the rolling window."""
        self._evict(current_ns)
        count = len(self._fill_times)
        if count == 0:
            return 0.0
        window_min = self.window_ns / 60_000_000_000
        return count / window_min

    def get_spread_adjustment(self, current_ns: int) -> float:
        """Returns multiplicative spread adjustment: >1 = widen, <1 = tighten."""
        rate = self.get_fill_rate(current_ns)
        if self.target_fill_rate <= 0:
            return 1.0
        ratio = rate / self.target_fill_rate
        return max(self.min_scale, min(self.max_scale, ratio))

    def _evict(self, current_ns: int) -> None:
        """Remove timestamps outside the rolling window."""
        cutoff = current_ns - self.window_ns
        while self._fill_times and self._fill_times[0] < cutoff:
            self._fill_times.popleft()
