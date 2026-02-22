"""
BasicMMSignal — inventory-controlled infinite horizon market making
with a pluggable volatility model.

The volatility model serves dual purposes:
1. Circuit breaker: stop quoting when predicted movement > threshold
2. AS sigma: the prediction is converted to per-tick sigma and used
   in the Avellaneda-Stoikov spread/reservation price formulas.
   Falls back to EWM volatility when the model returns None.
"""

import numpy as np
import pandas as pd

from gnomepy.backtest.recorder import BaseRecorder
from gnomepy.backtest.stats.stats import BaseRecord
from gnomepy.data.types import SchemaType
from gnomepy.registry.types import Listing
from gnomepy.research.signals.market_making import MarketMakingSignal
from gnomepy.research.types import Intent

from .volatility_model import VolatilityModel


class BasicMMRecord(BaseRecord):

    @classmethod
    def get_dtype(cls) -> np.dtype:
        return np.dtype([
            ('timestamp', 'i8'),
            ('mid_price', 'f8'),
            ('inventory', 'f8'),
            ('realized_vol', 'f8'),
            ('predicted_vol_bps', 'f8'),
            ('optimal_bid', 'f8'),
            ('optimal_ask', 'f8'),
            ('reservation_price', 'f8'),
            ('best_bid', 'f8'),
            ('best_ask', 'f8'),
            ('bid_confidence', 'f8'),
            ('ask_confidence', 'f8'),
            ('circuit_breaker_active', 'i1'),
        ])

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

        df['mid_price'] = df['mid_price'].ffill().bfill()
        df['realized_vol'] = df['realized_vol'].ffill().bfill()
        df['predicted_vol_bps'] = df['predicted_vol_bps'].ffill().bfill()
        df['optimal_bid'] = df['optimal_bid'].ffill().bfill()
        df['optimal_ask'] = df['optimal_ask'].ffill().bfill()
        df['reservation_price'] = df['reservation_price'].ffill().bfill()
        df['best_bid'] = df['best_bid'].ffill().bfill()
        df['best_ask'] = df['best_ask'].ffill().bfill()
        df['spread'] = df['best_ask'] - df['best_bid']
        df['inventory'] = df['inventory'].ffill().fillna(0.0)

        return df


class BasicMMRecorder(BaseRecorder):

    def get_record_class(self):
        return BasicMMRecord

    def log(
        self,
        listing_id: int,
        timestamp: int,
        mid_price: float,
        inventory: float,
        realized_vol: float,
        predicted_vol_bps: float,
        optimal_bid: float,
        optimal_ask: float,
        reservation_price: float,
        best_bid: float,
        best_ask: float,
        bid_confidence: float,
        ask_confidence: float,
        circuit_breaker_active: bool,
    ):
        asset_no = self.listing_id_to_asset_no[listing_id]
        i = self.at_i[asset_no]

        if i >= len(self.records):
            if self.auto_resize:
                self._resize_buffer()
            else:
                raise IndexError(f"Buffer full for asset {listing_id}. Consider increasing size or enabling auto_resize.")

        self.records[i, asset_no]['timestamp'] = timestamp
        self.records[i, asset_no]['mid_price'] = mid_price
        self.records[i, asset_no]['inventory'] = inventory
        self.records[i, asset_no]['realized_vol'] = realized_vol
        self.records[i, asset_no]['predicted_vol_bps'] = predicted_vol_bps
        self.records[i, asset_no]['optimal_bid'] = optimal_bid
        self.records[i, asset_no]['optimal_ask'] = optimal_ask
        self.records[i, asset_no]['reservation_price'] = reservation_price
        self.records[i, asset_no]['best_bid'] = best_bid
        self.records[i, asset_no]['best_ask'] = best_ask
        self.records[i, asset_no]['bid_confidence'] = bid_confidence
        self.records[i, asset_no]['ask_confidence'] = ask_confidence
        self.records[i, asset_no]['circuit_breaker_active'] = int(circuit_breaker_active)

        self.at_i[asset_no] += 1


class BasicMMSignal(MarketMakingSignal):
    """Basic inventory-controlled market making with pluggable volatility.

    Uses the Avellaneda-Stoikov infinite horizon formulas for reservation price
    and optimal spread. The pluggable VolatilityModel provides:

    1. **AS sigma**: its prediction (bps over a horizon) is converted to
       per-tick sigma via ``predicted_bps / 1e4 / sqrt(horizon)`` and used
       as the volatility input to the AS formulas.
    2. **Circuit breaker**: when predicted movement exceeds
       ``vol_threshold_bps``, quoting stops entirely.

    When the vol model returns ``None`` (insufficient data early in a
    backtest), an internal EWM volatility is used as a fallback.
    """

    def __init__(
        self,
        listing: Listing,
        volatility_model: VolatilityModel,
        vol_threshold_bps: float = 10.0,
        data_schema_type: SchemaType = SchemaType.MBP_10,
        trade_frequency: int = 1,
        gamma: float = 0.1,
        order_arrival_rate: float = 1.0,
        volatility_window: int = 100,
        volatility_half_life: float = 0.5,
        max_inventory: float | None = None,
        liquidation_threshold: float = 0.8,
        use_market_orders_for_liquidation: bool = True,
        min_spread_bps: float | None = None,
        max_spread_ticks: float | None = None,
        min_volatility: float = 1e-8,
        recorder: "BasicMMRecorder | None" = None,
    ):
        super().__init__(
            listing=listing,
            data_schema_type=data_schema_type,
            trade_frequency=trade_frequency,
            max_lookback=max(volatility_window, volatility_model.min_lookback),
            max_inventory=max_inventory,
            liquidation_threshold=liquidation_threshold,
            use_market_orders_for_liquidation=use_market_orders_for_liquidation,
        )

        self.volatility_model = volatility_model
        self.vol_threshold_bps = vol_threshold_bps
        self.gamma = gamma
        self.order_arrival_rate = order_arrival_rate
        self.volatility_window = volatility_window
        self.volatility_half_life = volatility_half_life * volatility_window
        self.min_spread_bps = min_spread_bps
        self.max_spread_ticks = max_spread_ticks
        self.min_volatility = min_volatility
        self.recorder = recorder

    def _calculate_ewm_volatility(self, prices: np.ndarray) -> float:
        if len(prices) < 2:
            return 0.0
        log_returns = np.diff(np.log(prices))
        if len(log_returns) == 0:
            return 0.0
        return pd.Series(log_returns).ewm(span=self.volatility_window).std().iloc[-1]

    def generate_intents(
        self,
        data: dict[int, dict[str, np.ndarray]],
        positions: dict[int, float] = None,
        timestamp: int = None,
    ) -> list[Intent]:
        listing_id = self.listing.listing_id
        listing_data = data[listing_id]

        bid_prices = listing_data['bidPrice0']
        ask_prices = listing_data['askPrice0']
        current_bid = bid_prices[-1]
        current_ask = ask_prices[-1]

        # Check minimum data for EWM convergence
        min_data_points = max(2, int(self.volatility_half_life))
        if len(bid_prices) < min_data_points or len(ask_prices) < min_data_points:
            return []

        # Calculate recent mid prices
        lookback_size = min(len(bid_prices), self.max_lookback)
        recent_mids = (bid_prices[-lookback_size:] + ask_prices[-lookback_size:]) / 2.0
        current_mid = recent_mids[-1]

        # Volatility model prediction (used for both AS sigma and circuit breaker)
        predicted_vol_bps = self.volatility_model.predict(listing_data)

        if predicted_vol_bps is not None:
            # Convert predicted bps over horizon → per-tick sigma (log-return units)
            volatility = (predicted_vol_bps / 1e4) / np.sqrt(self.volatility_model.horizon)
        else:
            # Fallback: EWM volatility when vol model has insufficient data
            volatility = self._calculate_ewm_volatility(recent_mids)
        volatility = max(volatility, self.min_volatility)

        # Circuit breaker
        circuit_breaker_active = False

        if predicted_vol_bps is not None and predicted_vol_bps > self.vol_threshold_bps:
            circuit_breaker_active = True
            if self.recorder is not None and timestamp is not None:
                inventory = positions.get(listing_id, 0.0) if positions else 0.0
                self.recorder.log(
                    listing_id=listing_id,
                    timestamp=timestamp,
                    mid_price=current_mid,
                    inventory=inventory,
                    realized_vol=volatility,
                    predicted_vol_bps=predicted_vol_bps,
                    optimal_bid=np.nan,
                    optimal_ask=np.nan,
                    reservation_price=np.nan,
                    best_bid=current_bid,
                    best_ask=current_ask,
                    bid_confidence=0.0,
                    ask_confidence=0.0,
                    circuit_breaker_active=True,
                )
            return []

        # Get current inventory
        inventory = positions.get(listing_id, 0.0) if positions else 0.0

        # --- AS infinite horizon formulas ---

        # Reservation price
        if self.order_arrival_rate > 0:
            base_adjustment = self.gamma * (volatility ** 2) / (2.0 * self.order_arrival_rate)

            inventory_factor = abs(inventory)
            if self.max_inventory is not None and self.max_inventory > 0:
                normalized_inventory = inventory_factor / self.max_inventory
                non_linear_multiplier = 1.0 + (normalized_inventory ** 2)
            else:
                non_linear_multiplier = 1.0

            inventory_adjustment = inventory * base_adjustment * non_linear_multiplier
            reservation_price = current_mid - inventory_adjustment
        else:
            reservation_price = current_mid

        # Optimal spread: s = gamma * sigma^2 / k + (2/gamma) * ln(1 + gamma/k)
        spread_component_1 = 0.0
        spread_component_2 = 0.0

        if self.order_arrival_rate > 0 and volatility > 0:
            spread_component_1 = self.gamma * (volatility ** 2) / self.order_arrival_rate

            if self.gamma > 0:
                log_argument = 1.0 + self.gamma / self.order_arrival_rate
                if log_argument > 0:
                    spread_component_2 = (2.0 / self.gamma) * np.log(log_argument)
                else:
                    spread_component_2 = 0.0

        optimal_spread = max(0.0, spread_component_1 + spread_component_2)

        # Check if near max inventory (aggressive liquidation mode)
        near_max_inventory = False
        if self.max_inventory is not None:
            liquidation_threshold_inventory = self.max_inventory * self.liquidation_threshold
            near_max_inventory = abs(inventory) >= liquidation_threshold_inventory

        # Apply minimum spread in bps (only during normal market making)
        if self.min_spread_bps is not None and not near_max_inventory and current_mid > 0:
            min_spread_absolute = current_mid * (self.min_spread_bps / 10000.0)
            optimal_spread = max(optimal_spread, min_spread_absolute)

        # Apply spread cap based on volatility
        if self.max_spread_ticks is not None and volatility > 0:
            max_spread = self.max_spread_ticks * volatility
            optimal_spread = min(optimal_spread, max_spread)

        optimal_bid = reservation_price - optimal_spread / 2.0
        optimal_ask = reservation_price + optimal_spread / 2.0

        # Cap prices to avoid crossing spread
        if not near_max_inventory:
            optimal_bid = min(optimal_bid, current_mid)
            optimal_ask = max(optimal_ask, current_mid)
        else:
            optimal_bid = min(optimal_bid, current_ask)
            optimal_ask = max(optimal_ask, current_bid)

        # --- Generate intents ---
        intents = []

        if self.max_inventory is None or inventory < self.max_inventory:
            if current_bid > 0 and optimal_bid > current_bid:
                bid_improvement_pct = (optimal_bid - current_bid) / current_bid
                bid_confidence = min(bid_improvement_pct * 50.0, 1.0)
            else:
                bid_confidence = 0.1
            bid_confidence = max(bid_confidence, 0.1)

            intents.append(Intent(
                listing=self.listing,
                side="B",
                confidence=bid_confidence,
                price=optimal_bid,
            ))
        else:
            bid_confidence = 0.0

        if self.max_inventory is None or inventory > -self.max_inventory:
            if current_ask > 0 and optimal_ask < current_ask:
                ask_improvement_pct = (current_ask - optimal_ask) / current_ask
                ask_confidence = min(ask_improvement_pct * 50.0, 1.0)
            else:
                ask_confidence = 0.1
            ask_confidence = max(ask_confidence, 0.1)

            intents.append(Intent(
                listing=self.listing,
                side="S",
                confidence=ask_confidence,
                price=optimal_ask,
            ))
        else:
            ask_confidence = 0.0

        if self.recorder is not None and timestamp is not None:
            self.recorder.log(
                listing_id=listing_id,
                timestamp=timestamp,
                mid_price=current_mid,
                inventory=inventory,
                realized_vol=volatility,
                predicted_vol_bps=predicted_vol_bps if predicted_vol_bps is not None else np.nan,
                optimal_bid=optimal_bid,
                optimal_ask=optimal_ask,
                reservation_price=reservation_price,
                best_bid=current_bid,
                best_ask=current_ask,
                bid_confidence=bid_confidence,
                ask_confidence=ask_confidence,
                circuit_breaker_active=False,
            )

        return intents
