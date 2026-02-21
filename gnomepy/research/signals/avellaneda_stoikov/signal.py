from gnomepy.data.types import SchemaType
from gnomepy.registry.types import Listing
from gnomepy.research.signals.market_making import MarketMakingSignal
from gnomepy.research.types import Intent
import numpy as np
import pandas as pd
from gnomepy.backtest.stats.stats import BaseRecord
from gnomepy.backtest.recorder import BaseRecorder


class AvellanedaStoikovModelRecord(BaseRecord):

    @classmethod
    def get_dtype(cls) -> np.dtype:
        return np.dtype([
            ('timestamp', 'i8'),
            ('mid_price', 'f8'),
            ('inventory', 'f8'),
            ('volatility', 'f8'),
            ('optimal_bid', 'f8'),
            ('optimal_ask', 'f8'),
            ('reservation_price', 'f8'),
            ('best_bid', 'f8'),
            ('best_ask', 'f8'),
            ('bid_confidence', 'f8'),
            ('ask_confidence', 'f8'),
        ])

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame without mutating the original."""
        df = df.copy()

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

        df['mid_price'] = df['mid_price'].ffill().bfill()
        df['volatility'] = df['volatility'].ffill().bfill()
        df['optimal_bid'] = df['optimal_bid'].ffill().bfill()
        df['optimal_ask'] = df['optimal_ask'].ffill().bfill()
        df['reservation_price'] = df['reservation_price'].ffill().bfill()
        df['best_bid'] = df['best_bid'].ffill().bfill()
        df['best_ask'] = df['best_ask'].ffill().bfill()
        df['spread'] = df['best_ask'] - df['best_bid']
        df['inventory'] = df['inventory'].ffill().fillna(0.0)

        return df

class AvellanedaStoikovModelValueRecorder(BaseRecorder):

    def get_record_class(self):
        return AvellanedaStoikovModelRecord

    def log(self, listing_id: int, timestamp: int, mid_price: float, inventory: float, volatility: float, optimal_bid: float, optimal_ask: float, reservation_price: float, best_bid: float, best_ask: float, bid_confidence: float, ask_confidence: float):
        """Log a model value record."""
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
        self.records[i, asset_no]['volatility'] = volatility
        self.records[i, asset_no]['optimal_bid'] = optimal_bid
        self.records[i, asset_no]['optimal_ask'] = optimal_ask
        self.records[i, asset_no]['reservation_price'] = reservation_price
        self.records[i, asset_no]['best_bid'] = best_bid
        self.records[i, asset_no]['best_ask'] = best_ask
        self.records[i, asset_no]['bid_confidence'] = bid_confidence
        self.records[i, asset_no]['ask_confidence'] = ask_confidence

        self.at_i[asset_no] += 1


class AvellanedaStoikovSignal(MarketMakingSignal):
    """Avellaneda-Stoikov optimal market making algorithm.

    Implements the optimal market making strategy from:
    Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order market.

    Uses infinite horizon formulation (time-independent). Both volatility and
    order_arrival_rate are in units of ticks.
    """

    def __init__(
            self,
            listing: Listing,
            data_schema_type: SchemaType = SchemaType.MBP_10,
            trade_frequency: int = 1,
            volatility_window: int = 100,
            volatility_half_life: float = 0.5,
            gamma: float = 0.1,
            order_arrival_rate: float = 1.0,
            max_inventory: float = None,
            liquidation_threshold: float = 0.8,
            use_market_orders_for_liquidation: bool = True,
            recorder: AvellanedaStoikovModelValueRecorder | None = None,
            max_spread_ticks: float = None,
            min_volatility: float = 1e-8,
            min_spread_bps: float = None,
    ):
        super().__init__(
            listing=listing,
            data_schema_type=data_schema_type,
            trade_frequency=trade_frequency,
            max_lookback=volatility_window,
            max_inventory=max_inventory,
            liquidation_threshold=liquidation_threshold,
            use_market_orders_for_liquidation=use_market_orders_for_liquidation,
        )

        self.gamma = gamma
        self.order_arrival_rate = order_arrival_rate
        self.max_spread_ticks = max_spread_ticks
        self.min_volatility = min_volatility
        self.min_spread_bps = min_spread_bps
        self.volatility_window = volatility_window
        self.volatility_half_life = volatility_half_life * volatility_window
        self.recorder = recorder

    def _calculate_mid_price(self, bid_prices: np.ndarray, ask_prices: np.ndarray) -> np.ndarray:
        return (bid_prices + ask_prices) / 2.0

    def _calculate_ewm_volatility(self, prices: np.ndarray, window: int = 100) -> float:
        if len(prices) < 2:
            return 0.0
        log_returns = np.diff(np.log(prices))
        if len(log_returns) == 0:
            return 0.0
        return pd.Series(log_returns).ewm(span=window).std().iloc[-1]

    def calculate_optimal_prices(
            self,
            timestamp: int,
            mid_price: float,
            inventory: float,
            volatility: float,
            best_bid: float = None,
            best_ask: float = None,
    ) -> tuple[float, float, float]:
        """Calculate optimal bid and ask prices using Avellaneda-Stoikov infinite horizon formula.

        Returns:
            Tuple of (optimal_bid, optimal_ask, reservation_price)
        """
        if self.order_arrival_rate > 0:
            base_adjustment = self.gamma * (volatility ** 2) / (2.0 * self.order_arrival_rate)

            inventory_factor = abs(inventory)
            if self.max_inventory is not None and self.max_inventory > 0:
                normalized_inventory = inventory_factor / self.max_inventory
                non_linear_multiplier = 1.0 + (normalized_inventory ** 2)
            else:
                non_linear_multiplier = 1.0

            inventory_adjustment = inventory * base_adjustment * non_linear_multiplier
            reservation_price = mid_price - inventory_adjustment
        else:
            reservation_price = mid_price

        # Infinite horizon optimal spread: s = gamma * sigma^2 / k + (2/gamma) * ln(1 + gamma/k)
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

        # Check if we're near max inventory (aggressive liquidation mode)
        near_max_inventory = False
        if self.max_inventory is not None:
            liquidation_threshold_inventory = self.max_inventory * self.liquidation_threshold
            near_max_inventory = abs(inventory) >= liquidation_threshold_inventory

        # Apply minimum spread in bps (only during normal market making, not high inventory)
        if self.min_spread_bps is not None and not near_max_inventory and mid_price > 0:
            min_spread_absolute = mid_price * (self.min_spread_bps / 10000.0)
            optimal_spread = max(optimal_spread, min_spread_absolute)

        # Apply spread cap based on volatility (if configured)
        if self.max_spread_ticks is not None and volatility > 0:
            max_spread = self.max_spread_ticks * volatility
            optimal_spread = min(optimal_spread, max_spread)

        optimal_bid = reservation_price - optimal_spread / 2.0
        optimal_ask = reservation_price + optimal_spread / 2.0

        # Cap prices to avoid crossing spread during normal market making
        if best_bid is not None and best_ask is not None:
            if not near_max_inventory:
                optimal_bid = min(optimal_bid, mid_price)
                optimal_ask = max(optimal_ask, mid_price)
            else:
                optimal_bid = min(optimal_bid, best_ask)
                optimal_ask = max(optimal_ask, best_bid)

        return optimal_bid, optimal_ask, reservation_price

    def generate_intents(
            self,
            data: dict[int, dict[str, np.ndarray]],
            positions: dict[int, float] = None,
            timestamp: int = None
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
        recent_mids = self._calculate_mid_price(bid_prices[-lookback_size:], ask_prices[-lookback_size:])
        current_mid = recent_mids[-1]

        # Calculate exponentially weighted volatility per tick
        volatility = self._calculate_ewm_volatility(recent_mids, window=self.volatility_window)
        volatility = max(volatility, self.min_volatility)

        # Get current inventory
        inventory = positions.get(listing_id, 0.0) if positions else 0.0

        # Calculate optimal prices
        optimal_bid, optimal_ask, reservation_price = self.calculate_optimal_prices(
            timestamp=timestamp,
            mid_price=current_mid,
            inventory=inventory,
            volatility=volatility,
            best_bid=current_bid,
            best_ask=current_ask
        )

        # Generate intents
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
                price=optimal_bid
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
                price=optimal_ask
            ))
        else:
            ask_confidence = 0.0

        if self.recorder is not None:
            self.recorder.log(
                listing_id=self.listing.listing_id,
                timestamp=timestamp,
                mid_price=current_mid,
                inventory=inventory,
                volatility=volatility,
                optimal_bid=optimal_bid,
                optimal_ask=optimal_ask,
                reservation_price=reservation_price,
                best_bid=current_bid,
                best_ask=current_ask,
                bid_confidence=bid_confidence,
                ask_confidence=ask_confidence,
            )

        return intents
