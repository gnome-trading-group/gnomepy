from __future__ import annotations


class L2Book:
    """Reconstructs a top-N order book from incremental L2 updates.

    Mirrors the depth computation in the Java MbpBufferBook/Mbp10Book gateways:
    depth = the shallowest (minimum) level index where the top-N changed.
    Returns None when the update falls entirely outside the top-N visible levels.
    """

    NUM_LEVELS = 10

    def __init__(self) -> None:
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self._top_bids: list[tuple[float, float]] = []
        self._top_asks: list[tuple[float, float]] = []

    def clear(self) -> None:
        self._bids.clear()
        self._asks.clear()
        self._top_bids = []
        self._top_asks = []

    def update(self, side: str, price: float, amount: float) -> int | None:
        """Apply one L2 update. Returns depth if the top-N changed, else None."""
        book = self._bids if side == "bid" else self._asks
        if amount == 0.0:
            book.pop(price, None)
        else:
            book[price] = amount

        new_bids = sorted(self._bids.items(), reverse=True)[: self.NUM_LEVELS]
        new_asks = sorted(self._asks.items())[: self.NUM_LEVELS]

        depth = self._shallowest_change(self._top_bids, new_bids, self._top_asks, new_asks)

        self._top_bids = new_bids
        self._top_asks = new_asks
        return depth

    def top_levels(self) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Return (top_bids, top_asks) as lists of (price, amount), sorted best-first."""
        return self._top_bids, self._top_asks

    def _shallowest_change(
        self,
        prev_bids: list[tuple[float, float]],
        new_bids: list[tuple[float, float]],
        prev_asks: list[tuple[float, float]],
        new_asks: list[tuple[float, float]],
    ) -> int | None:
        for i in range(self.NUM_LEVELS):
            prev_bid = prev_bids[i] if i < len(prev_bids) else None
            new_bid = new_bids[i] if i < len(new_bids) else None
            prev_ask = prev_asks[i] if i < len(prev_asks) else None
            new_ask = new_asks[i] if i < len(new_asks) else None
            if prev_bid != new_bid or prev_ask != new_ask:
                return i
        return None
