"""
Runtime signal for return prediction.

Extends ``PositionAwareSignal`` directly (not ``MarketMakingSignal``)
since this is a directional strategy that uses market orders, not a
quoting strategy.
"""

import numpy as np
import pandas as pd

from gnomepy.backtest.recorder import BaseRecorder
from gnomepy.backtest.stats.stats import BaseRecord
from gnomepy.data.types import SchemaType
from gnomepy.registry.types import Listing
from gnomepy.research.models.bps import BpsModel
from gnomepy.research.signals import PositionAwareSignal
from gnomepy.research.types import Intent


class BpsRecord(BaseRecord):

    @classmethod
    def get_dtype(cls) -> np.dtype:
        return np.dtype([
            ('timestamp', 'i8'),
            ('mid_price', 'f8'),
            ('predicted_return', 'f8'),
            ('inventory', 'f8'),
            ('pnl_bps', 'f8'),
            ('ticks_in_position', 'i8'),
            ('entry_price', 'f8'),
        ])

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

        df['mid_price'] = df['mid_price'].ffill().bfill()
        df['inventory'] = df['inventory'].ffill().fillna(0.0)
        df['ticks_in_position'] = df['ticks_in_position'].ffill().fillna(0)

        return df


class BpsRecorder(BaseRecorder):

    def get_record_class(self):
        return BpsRecord

    def log(self, listing_id: int, timestamp: int, mid_price: float,
            predicted_return: float, inventory: float, pnl_bps: float,
            ticks_in_position: int, entry_price: float):
        asset_no = self.listing_id_to_asset_no[listing_id]
        i = self.at_i[asset_no]

        if i >= len(self.records):
            if self.auto_resize:
                self._resize_buffer()
            else:
                raise IndexError(f"Buffer full for asset {listing_id}. Consider increasing size or enabling auto_resize.")

        self.records[i, asset_no]['timestamp'] = timestamp
        self.records[i, asset_no]['mid_price'] = mid_price
        self.records[i, asset_no]['predicted_return'] = predicted_return
        self.records[i, asset_no]['inventory'] = inventory
        self.records[i, asset_no]['pnl_bps'] = pnl_bps
        self.records[i, asset_no]['ticks_in_position'] = ticks_in_position
        self.records[i, asset_no]['entry_price'] = entry_price

        self.at_i[asset_no] += 1


class BpsSignal(PositionAwareSignal):
    """Return prediction signal.

    Predicts short-term signed expected forward return from order book
    features and executes via market orders when the predicted return
    magnitude exceeds transaction cost.

    Parameters
    ----------
    listing : Listing
        The listing to trade.
    model : BpsModel, optional
        Pre-constructed return model.
    trade_frequency : int
        Only generate intents every N ticks.
    entry_threshold_bps : float
        Minimum |predicted_return| in bps to enter a trade.
    transaction_cost_bps : float
        Round-trip transaction cost in basis points.
    max_inventory : float
        Maximum absolute position size (in base units).
    exit_hold_ticks : int
        Close position after holding for this many ticks.
    stop_loss_bps : float
        Stop-loss threshold in basis points.
    take_profit_bps : float
        Take-profit threshold in basis points.
    """

    def __init__(
        self,
        listing: Listing,
        model: BpsModel | None = None,
        trade_frequency: int = 1,
        entry_threshold_bps: float = 5.0,
        transaction_cost_bps: float = 2.0,
        max_inventory: float = 100.0,
        exit_hold_ticks: int = 100,
        stop_loss_bps: float = 10.0,
        take_profit_bps: float = 15.0,
        recorder: BpsRecorder | None = None,
    ):
        super().__init__()

        self.model: BpsModel | None = model

        # OMS-required attributes
        self.listing = listing
        self.listings = [listing]
        self.data_schema_type = SchemaType.MBP_10
        self.trade_frequency = trade_frequency
        self.max_lookback = self.model.min_lookback if self.model is not None else 201
        self.elapsed_ticks: dict[int, int] = {listing.listing_id: 0}

        # Signal parameters
        self.entry_threshold_bps = entry_threshold_bps
        self.transaction_cost_bps = transaction_cost_bps
        self.max_inventory = max_inventory
        self.exit_hold_ticks = exit_hold_ticks
        self.stop_loss_bps = stop_loss_bps
        self.take_profit_bps = take_profit_bps

        # Recorder
        self.recorder = recorder

        # Internal state
        self._entry_price: float | None = None
        self._position_side: str | None = None  # "B" or "S"
        self._ticks_since_entry: int = 0
        self._previous_inventory: float = 0.0
        self._entry_pending: bool = False
        self._exit_pending: bool = False

    # ------------------------------------------------------------------
    # OMS entry point
    # ------------------------------------------------------------------

    def process_new_tick(
        self,
        data: dict[int, dict[str, np.ndarray]],
        ticker_listing_id: int,
        positions: dict[int, float] | None = None,
        timestamp: int | None = None,
    ) -> list[Intent]:
        listing_id = self.listing.listing_id

        if ticker_listing_id != listing_id:
            return []
        if listing_id not in data:
            return []

        listing_data = data[listing_id]
        if "bidPrice0" not in listing_data or "askPrice0" not in listing_data:
            return []

        # Trade frequency gate
        if listing_id not in self.elapsed_ticks:
            self.elapsed_ticks[listing_id] = 0
        self.elapsed_ticks[listing_id] += 1
        if self.elapsed_ticks[listing_id] % self.trade_frequency != 0:
            return []

        return self._generate_intents(listing_data, positions, timestamp)

    # ------------------------------------------------------------------
    # Intent generation
    # ------------------------------------------------------------------

    def _generate_intents(
        self,
        listing_data: dict[str, np.ndarray],
        positions: dict[int, float] | None,
        timestamp: int | None = None,
    ) -> list[Intent]:
        listing_id = self.listing.listing_id
        inventory = positions.get(listing_id, 0.0) if positions else 0.0

        cur_bid = float(listing_data["bidPrice0"][-1])
        cur_ask = float(listing_data["askPrice0"][-1])
        cur_mid = (cur_bid + cur_ask) / 2.0

        # Any position change means a fill or rejection was processed
        if inventory != self._previous_inventory:
            self._entry_pending = False
            self._exit_pending = False

        # Detect position transitions
        if self._previous_inventory == 0.0 and inventory != 0.0:
            # 0 → non-zero: new position filled
            self._entry_price = cur_mid
            self._position_side = "B" if inventory > 0 else "S"
            self._ticks_since_entry = 0
        elif inventory == 0.0 and self._previous_inventory != 0.0:
            # non-zero → 0: position closed
            self._reset_state()
        elif abs(inventory) > 0:
            self._ticks_since_entry += 1

        self._previous_inventory = inventory

        predicted_return = np.nan  # default when model doesn't run

        # --- Exit checks (priority order) ---
        exit_intent = self._check_exits(inventory, cur_mid)
        if exit_intent is not None:
            if not self._exit_pending:
                self._exit_pending = True
                self._log_record(timestamp, cur_mid, predicted_return, inventory)
                return [exit_intent]
            self._log_record(timestamp, cur_mid, predicted_return, inventory)
            return []

        # --- Entry logic ---
        if self.model is None:
            self._log_record(timestamp, cur_mid, predicted_return, inventory)
            return []

        prediction = self.model.predict(listing_data)
        if prediction is None:
            self._log_record(timestamp, cur_mid, predicted_return, inventory)
            return []

        predicted_return = prediction
        abs_return = abs(predicted_return)

        intents: list[Intent] = []

        # Check for signal reversal exit while we have a position
        if abs(inventory) > 0:
            if not self._exit_pending:
                reversal = self._check_signal_reversal(inventory, predicted_return)
                if reversal is not None:
                    self._exit_pending = True
                    self._log_record(timestamp, cur_mid, predicted_return, inventory)
                    return [reversal]
            # Still holding — wait for an exit before entering again
            self._log_record(timestamp, cur_mid, predicted_return, inventory)
            return []

        # Don't generate new entries while waiting for a pending fill
        if self._entry_pending:
            self._log_record(timestamp, cur_mid, predicted_return, inventory)
            return []

        confidence = min(abs_return / (2 * self.transaction_cost_bps), 1.0)

        # Entry: long
        if (predicted_return > self.transaction_cost_bps
                and abs_return > self.entry_threshold_bps
                and (self.max_inventory is None or inventory < self.max_inventory)):
            intents.append(
                Intent(listing=self.listing, side="B", confidence=confidence, price=None)
            )

        # Entry: short
        elif (predicted_return < -self.transaction_cost_bps
              and abs_return > self.entry_threshold_bps
              and (self.max_inventory is None or inventory > -self.max_inventory)):
            intents.append(
                Intent(listing=self.listing, side="S", confidence=confidence, price=None)
            )

        if intents:
            self._entry_pending = True

        self._log_record(timestamp, cur_mid, predicted_return, inventory)
        return intents

    # ------------------------------------------------------------------
    # Exit checks
    # ------------------------------------------------------------------

    def _check_exits(self, inventory: float, cur_mid: float) -> Intent | None:
        """Check exit conditions in priority order.

        Returns an exit Intent or None.
        """
        if abs(inventory) == 0 or self._entry_price is None:
            return None

        # Unrealized PnL in bps
        if self._position_side == "B":
            pnl_bps = (cur_mid - self._entry_price) / self._entry_price * 1e4
        else:
            pnl_bps = (self._entry_price - cur_mid) / self._entry_price * 1e4

        # 1. Stop-loss
        if pnl_bps < -self.stop_loss_bps:
            return Intent(
                listing=self.listing,
                side="S" if inventory > 0 else "B",
                confidence=1.0,
                flatten=True,
                price=None,
            )

        # 2. Take-profit
        if pnl_bps > self.take_profit_bps:
            return Intent(
                listing=self.listing,
                side="S" if inventory > 0 else "B",
                confidence=1.0,
                flatten=True,
                price=None,
            )

        # 3. Time-based
        if self._ticks_since_entry >= self.exit_hold_ticks:
            return Intent(
                listing=self.listing,
                side="S" if inventory > 0 else "B",
                confidence=0.8,
                flatten=True,
                price=None,
            )

        return None

    def _check_signal_reversal(self, inventory: float, predicted_return: float) -> Intent | None:
        """Exit if model predicts opposite direction above entry threshold."""
        if inventory > 0 and predicted_return < -self.entry_threshold_bps:
            return Intent(
                listing=self.listing,
                side="S",
                confidence=0.9,
                flatten=True,
                price=None,
            )
        if inventory < 0 and predicted_return > self.entry_threshold_bps:
            return Intent(
                listing=self.listing,
                side="B",
                confidence=0.9,
                flatten=True,
                price=None,
            )
        return None

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _log_record(self, timestamp: int | None, cur_mid: float,
                    predicted_return: float, inventory: float) -> None:
        if self.recorder is None or timestamp is None:
            return

        # Compute unrealized PnL in bps (same formula as _check_exits)
        if self._entry_price is not None and abs(inventory) > 0:
            if self._position_side == "B":
                pnl_bps = (cur_mid - self._entry_price) / self._entry_price * 1e4
            else:
                pnl_bps = (self._entry_price - cur_mid) / self._entry_price * 1e4
        else:
            pnl_bps = np.nan

        self.recorder.log(
            listing_id=self.listing.listing_id,
            timestamp=timestamp,
            mid_price=cur_mid,
            predicted_return=predicted_return,
            inventory=inventory,
            pnl_bps=pnl_bps,
            ticks_in_position=self._ticks_since_entry,
            entry_price=self._entry_price if self._entry_price is not None else np.nan,
        )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self._entry_price = None
        self._position_side = None
        self._ticks_since_entry = 0
        self._previous_inventory = 0.0
        self._entry_pending = False
        self._exit_pending = False
