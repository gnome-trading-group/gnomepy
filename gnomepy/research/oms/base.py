from abc import ABC, abstractmethod
import dataclasses
import logging
import numpy as np

from gnomepy.data.types import SchemaBase, Order, OrderExecutionReport, FIXED_PRICE_SCALE, FIXED_SIZE_SCALE
from gnomepy.research.signals import Signal, PositionAwareSignal
from gnomepy.research.types import BasketIntent, Intent
from gnomepy.backtest.recorder import MarketRecorder, RecordType

logger = logging.getLogger(__name__)


class BaseOMS(ABC):
    """Abstract base for order management systems.

    Provides shared initialisation, listing lookups, market-data ingestion
    and intent generation.  Subclasses implement ``on_execution_report``
    and ``on_market_update`` to decide how intents become orders.
    """

    def __init__(self, signals: list[Signal], notional: float, starting_cash: float = 1000000.0):
        self.signals = signals
        self.notional = notional
        self.cash = starting_cash

        # Infer listings from signals
        all_listings = []
        for signal in signals:
            all_listings.extend(signal.listings)

        self.listing_data: dict[int, dict[str, np.ndarray]] = {}

        self.signal_positions: dict[Signal, dict[int, float]] = {}
        for signal in signals:
            self.signal_positions[signal] = {listing.listing_id: 0.0 for listing in signal.listings}

        self.positions: dict[int, float] = {listing.listing_id: 0.0 for listing in all_listings}
        self.order_log: dict[str, Order] = {}
        self._order_counter: int = 0
        self.elapsed_ticks: dict[int, int] = {listing.listing_id: 0 for listing in all_listings}

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _next_client_oid(self, prefix: str = "oms") -> str:
        self._order_counter += 1
        return f"{prefix}_{self._order_counter}"

    def _find_listing_id(self, exchange_id: int, security_id: int) -> int | None:
        """Resolve (exchange_id, security_id) → listing_id."""
        for signal in self.signals:
            for listing in signal.listings:
                if listing.exchange_id == exchange_id and listing.security_id == security_id:
                    return listing.listing_id
        return None

    def _find_listing(self, listing_id: int):
        """Find a Listing object by listing_id across all signals."""
        for signal in self.signals:
            for listing in signal.listings:
                if listing.listing_id == listing_id:
                    return listing
        return None

    # ------------------------------------------------------------------
    # Shared fill handling
    # ------------------------------------------------------------------

    def _apply_fill(
        self,
        listing_id: int,
        order: Order,
        execution_report: OrderExecutionReport,
        timestamp: int,
        recorder: MarketRecorder,
    ):
        """Update cash, positions and log an execution fill."""
        filled_qty = execution_report.filled_qty / FIXED_SIZE_SCALE
        filled_price = execution_report.filled_price / FIXED_PRICE_SCALE
        position_change = filled_qty if order.side == "B" else -filled_qty

        trade_value = filled_qty * filled_price if filled_price > 0 else 0.0
        if order.side == "B":
            self.cash -= trade_value
        else:
            self.cash += trade_value

        if listing_id in self.positions:
            self.positions[listing_id] += position_change

        # Update signal-specific positions
        signals_for_listing = [
            s for s in self.signals
            if any(l.listing_id == listing_id for l in s.listings)
        ]
        if signals_for_listing:
            position_change_per_signal = position_change / len(signals_for_listing)
            for signal in signals_for_listing:
                if listing_id in self.signal_positions[signal]:
                    self.signal_positions[signal][listing_id] += position_change_per_signal

        mid_price = execution_report.mid_price / FIXED_PRICE_SCALE
        recorder.log(
            event=RecordType.EXECUTION,
            listing_id=listing_id,
            timestamp=timestamp,
            price=mid_price if mid_price > 0 else filled_price,
            fill_price=filled_price,
            quantity=self.positions[listing_id],
            fee=execution_report.fee / (FIXED_PRICE_SCALE * FIXED_SIZE_SCALE),
        )

    # ------------------------------------------------------------------
    # Shared market-data processing
    # ------------------------------------------------------------------

    def _process_market_data(
        self,
        timestamp: int,
        market_update: SchemaBase,
        market_recorder: MarketRecorder,
    ) -> tuple[int | None, list]:
        """Ingest a market tick and generate signal intents.

        Returns ``(listing_id, all_intents)``.  If the listing is unknown
        returns ``(None, [])``.
        """
        listing_id = self._find_listing_id(market_update.exchange_id, market_update.security_id)
        if listing_id is None:
            return None, []

        self.elapsed_ticks[listing_id] += 1

        min_trade_frequency = min(
            getattr(signal, 'trade_frequency', 1) for signal in self.signals
        )
        should_append_data = (self.elapsed_ticks[listing_id] % min_trade_frequency == 0)

        if listing_id not in self.listing_data:
            self.listing_data[listing_id] = {}

        # Convert market data to dict and flatten levels
        market_dict = dataclasses.asdict(market_update)
        levels = market_dict.pop('levels', [])
        for i, level in enumerate(levels):
            market_dict[f'bidPrice{i}'] = level.get('bid_px', 0) / FIXED_PRICE_SCALE
            market_dict[f'askPrice{i}'] = level.get('ask_px', 0) / FIXED_PRICE_SCALE
            market_dict[f'bidSize{i}'] = level.get('bid_sz', 0) / FIXED_SIZE_SCALE
            market_dict[f'askSize{i}'] = level.get('ask_sz', 0) / FIXED_SIZE_SCALE
            market_dict[f'bidCount{i}'] = level.get('bid_ct', 0)
            market_dict[f'askCount{i}'] = level.get('ask_ct', 0)

        market_recorder.log_market_event(
            listing_id=listing_id,
            timestamp=timestamp,
            market_update=market_update,
            quantity=self.positions[listing_id],
        )

        # Append to numpy arrays
        if should_append_data:
            max_history_records = max(s.max_lookback for s in self.signals)
            for column, value in market_dict.items():
                if isinstance(value, str) or value is None:
                    continue
                try:
                    float_value = float(value)
                except (ValueError, TypeError):
                    continue

                if column not in self.listing_data[listing_id]:
                    self.listing_data[listing_id][column] = np.array([float_value], dtype=np.float64)
                else:
                    current_array = self.listing_data[listing_id][column]
                    new_array = np.append(current_array, float_value)
                    if len(new_array) > max_history_records:
                        new_array = new_array[-max_history_records:]
                    self.listing_data[listing_id][column] = new_array

        # Generate intents from all signals
        all_intents: list = []
        for signal in self.signals:
            if isinstance(signal, PositionAwareSignal):
                new_intents = signal.process_new_tick(
                    data=self.listing_data,
                    positions=self.signal_positions[signal],
                    ticker_listing_id=listing_id,
                    timestamp=timestamp,
                )
            else:
                new_intents = signal.process_new_tick(data=self.listing_data)

            if new_intents and len(new_intents) > 0:
                all_intents.extend(new_intents)
                for intent in new_intents:
                    if isinstance(intent, BasketIntent):
                        for sub_intent, proportion in zip(intent.intents, intent.proportions):
                            logger.debug(
                                f"Intent generated: listing={sub_intent.listing.listing_id}, "
                                f"side={sub_intent.side}, confidence={sub_intent.confidence * proportion:.3f}, "
                                f"price={sub_intent.price}, flatten={getattr(sub_intent, 'flatten', False)}"
                            )
                    else:
                        logger.debug(
                            f"Intent generated: listing={intent.listing.listing_id}, "
                            f"side={intent.side}, confidence={intent.confidence:.3f}, "
                            f"price={intent.price}, flatten={getattr(intent, 'flatten', False)}"
                        )

        return listing_id, all_intents

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def on_execution_report(self, timestamp: int, execution_report: OrderExecutionReport, recorder: MarketRecorder):
        ...

    @abstractmethod
    def on_market_update(self, timestamp: int, market_update: SchemaBase, market_recorder: MarketRecorder):
        ...
