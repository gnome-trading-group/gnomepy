from gnomepy.data.types import SchemaType
from gnomepy.registry.types import Listing
from gnomepy.research.signals.market_making import MarketMakingSignal
from gnomepy.research.signals._mixins import VolatilityMixin, PricingMixin
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
            ('spread', 'f8'),
            ('best_bid', 'f8'),
            ('best_ask', 'f8'),
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

    def log(self, listing_id: int, timestamp: int, mid_price: float, inventory: float, volatility: float, optimal_bid: float, optimal_ask: float, reservation_price: float, spread: float, best_bid: float, best_ask: float):
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
        self.records[i, asset_no]['spread'] = spread
        self.records[i, asset_no]['best_bid'] = best_bid
        self.records[i, asset_no]['best_ask'] = best_ask

        self.at_i[asset_no] += 1


class AvellanedaStoikovSignal(MarketMakingSignal, VolatilityMixin, PricingMixin):
    """Avellaneda-Stoikov optimal market making algorithm.

    This signal implements the optimal market making strategy from:
    Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order market.
    Journal of Financial Markets, 11(1), 1-31.

    The algorithm calculates optimal bid and ask prices based on:
    - Current inventory (position)
    - Risk aversion parameter (gamma)
    - Volatility of the asset (per tick)
    - Order arrival rate parameter (k, in orders per tick)

    Uses infinite horizon formulation (time-independent).

    Note: Both volatility and order_arrival_rate are in units of ticks.
    """

    def __init__(
            self,
            listing: Listing,
            data_schema_type: SchemaType = SchemaType.MBP_10,
            trade_frequency: int = 1,
            volatility_window: int = 100,
            volatility_span: float = None,
            gamma: float = 0.1,
            time_horizon: float = 1.0,
            order_arrival_rate: float = 1.0,
            max_inventory: float = None,
            liquidation_threshold: float = 0.8,
            use_market_orders_for_liquidation: bool = True,
            model_value_recorder: 'AvellanedaStoikovModelValueRecorder | None' = None,
    ):
        """Initialize an Avellaneda-Stoikov market making strategy.

        Parameters
        ----------
        listing : Listing
            Single listing to market make
        data_schema_type : SchemaType, default SchemaType.MBP_10
            Type of market data schema to use
        trade_frequency : int, default 1
            How frequently to check for trading signals
        volatility_window : int, default 100
            DEPRECATED: Use volatility_span instead. Kept for backward compatibility.
            If volatility_span is None, will be used to set volatility_span.
        volatility_span : float, optional
            Span parameter for exponentially weighted volatility calculation.
            If None, defaults to volatility_window.
            Higher values give more weight to older data (smoother).
        gamma : float, default 0.1
            Risk aversion parameter (higher = more risk averse)
        time_horizon : float, default 1.0
            DEPRECATED: Not used in infinite horizon formulation. Kept for backward compatibility.
        order_arrival_rate : float, default 1.0
            Order arrival rate parameter (k in the formula), in orders per tick
        max_inventory : float, optional
            Maximum absolute inventory size before stopping market making
        liquidation_threshold : float, default 0.8
            Fraction of max_inventory at which to start aggressive liquidation (0.0-1.0)
        use_market_orders_for_liquidation : bool, default True
            If True, use market orders when at max inventory for immediate liquidation
        """
        super().__init__(
            listing=listing,
            data_schema_type=data_schema_type,
            trade_frequency=trade_frequency,
            volatility_window=volatility_window,
            volatility_span=volatility_span,
            max_inventory=max_inventory,
            liquidation_threshold=liquidation_threshold,
            use_market_orders_for_liquidation=use_market_orders_for_liquidation,
        )

        self.gamma = gamma
        self.time_horizon = time_horizon  # Kept for backward compatibility but not used
        self.order_arrival_rate = order_arrival_rate
        self.model_value_recorder = model_value_recorder or AvellanedaStoikovModelValueRecorder([listing.listing_id])
        print(f"Initialized AvellanedaStoikovSignal for listing {listing.listing_id} (infinite horizon, EWM volatility)")

    def calculate_optimal_prices(
            self,
            timestamp: int,
            mid_price: float,
            inventory: float,
            volatility: float,
            best_bid: float = None,
            best_ask: float = None,
    ) -> tuple[float, float]:
        """Calculate optimal bid and ask prices using Avellaneda-Stoikov infinite horizon formula.

        Args:
            timestamp: Timestamp of the calculation
            mid_price: Current mid price
            inventory: Current inventory (position size)
            volatility: Volatility (sigma)
            best_bid: Current best bid price in the market (optional, for logging)
            best_ask: Current best ask price in the market (optional, for logging)

        Returns:
            Tuple of (optimal_bid, optimal_ask)
        """
        if self.order_arrival_rate > 0:
            # Base adjustment
            base_adjustment = self.gamma * (volatility ** 2) / (2.0 * self.order_arrival_rate)

            # Non-linear: stronger adjustment for larger inventory
            # Use abs(inventory) to make it symmetric
            inventory_factor = abs(inventory)
            if self.max_inventory is not None and self.max_inventory > 0:
                # Normalize by max_inventory and apply non-linear scaling
                normalized_inventory = inventory_factor / self.max_inventory
                # Quadratic scaling: adjustment grows faster with inventory
                non_linear_multiplier = 1.0 + (normalized_inventory ** 2)
            else:
                non_linear_multiplier = 1.0

            inventory_adjustment = inventory * base_adjustment * non_linear_multiplier
            reservation_price = mid_price - inventory_adjustment
        else:
            # Fallback if order_arrival_rate is zero or invalid
            reservation_price = mid_price

        # Infinite horizon optimal spread: s = gamma * sigma^2 / k + (2/gamma) * ln(1 + gamma/k)
        # where sigma is volatility per tick and k is order arrival rate (orders per tick)
        spread_component_1 = 0.0
        spread_component_2 = 0.0

        if self.order_arrival_rate > 0 and volatility > 0:
            spread_component_1 = self.gamma * (volatility ** 2) / self.order_arrival_rate

            # Avoid division by zero and log of non-positive
            if self.gamma > 0:
                # Ensure argument to log is positive
                log_argument = 1.0 + self.gamma / self.order_arrival_rate
                if log_argument > 0:
                    spread_component_2 = (2.0 / self.gamma) * np.log(log_argument)
                else:
                    # Fallback if log argument is invalid
                    spread_component_2 = 0.0

        optimal_spread = spread_component_1 + spread_component_2

        # Ensure spread is non-negative
        optimal_spread = max(0.0, optimal_spread)

        # Optimal bid and ask
        optimal_bid = reservation_price - optimal_spread / 2.0
        optimal_ask = reservation_price + optimal_spread / 2.0

        # Cap prices to avoid crossing spread during normal market making
        # But allow crossing when inventory is high (aggressive liquidation mode)
        if best_bid is not None and best_ask is not None:
            # Check if we're near max inventory (aggressive liquidation mode)
            near_max_inventory = False
            if self.max_inventory is not None:
                liquidation_threshold_inventory = self.max_inventory * self.liquidation_threshold
                near_max_inventory = abs(inventory) >= liquidation_threshold_inventory
            
            if not near_max_inventory:
                # Normal market making: cap at mid price to avoid paying spread
                optimal_bid = min(optimal_bid, mid_price)
                optimal_ask = max(optimal_ask, mid_price)
            # If near_max_inventory, allow crossing to aggressively reduce inventory

        if self.model_value_recorder is not None:
            # Use provided best_bid/best_ask or fallback to mid_price if not provided
            log_best_bid = best_bid if best_bid is not None else mid_price
            log_best_ask = best_ask if best_ask is not None else mid_price
            
            self.model_value_recorder.log(
                listing_id=self.listing.listing_id,
                timestamp=timestamp,
                mid_price=mid_price,
                inventory=inventory,
                volatility=volatility,
                optimal_bid=optimal_bid,
                optimal_ask=optimal_ask,
                reservation_price=reservation_price,
                spread=optimal_spread,
                best_bid=log_best_bid,
                best_ask=log_best_ask
            )


        return optimal_bid, optimal_ask

    def generate_intents(
            self,
            data: dict[int, dict[str, np.ndarray]],
            positions: dict[int, float] = None,
            timestamp: int = None
    ) -> list[Intent]:
        """Generate trading intents based on optimal prices from calculate_optimal_prices.

        Args:
            data: Dictionary mapping listing_id to their historical data (numpy arrays)
            positions: Dictionary mapping listing_id to their current positions
            timestamp: Optional timestamp for logging model values

        Returns:
            list: List of Intent objects
        """
        listing_id = self.listing.listing_id

        # Check if we have enough data
        if listing_id not in data:
            return []

        listing_data = data[listing_id]

        # Check if we have required price data
        if 'bidPrice0' not in listing_data or 'askPrice0' not in listing_data:
            return []

        bid_prices = listing_data['bidPrice0']
        ask_prices = listing_data['askPrice0']
        current_bid = bid_prices[-1]
        current_ask = ask_prices[-1]

        # For EWM volatility, we need at least 2 prices (for 1 return)
        # But use more data for better EWM convergence
        min_data_points = max(2, int(self.volatility_span * 0.5))  # At least half the span
        if len(bid_prices) < min_data_points or len(ask_prices) < min_data_points:
            return []

        # Calculate recent mid prices
        lookback_size = min(len(bid_prices), self.max_lookback)
        recent_mids = self.calculate_mid_price(bid_prices[-lookback_size:], ask_prices[-lookback_size:])

        current_mid = recent_mids[-1]

        # Calculate exponentially weighted volatility per tick
        volatility = self.calculate_ewm_volatility(recent_mids, window=self.volatility_window)

        # Avoid issues with zero or very small volatility
        if volatility < 1e-8:
            volatility = 1e-8

        # Get current inventory
        inventory = positions.get(listing_id, 0.0) if positions else 0.0

        # Calculate optimal bid and ask prices using the subclass implementation
        optimal_bid, optimal_ask = self.calculate_optimal_prices(
            timestamp=timestamp,
            mid_price=current_mid,
            inventory=inventory,
            volatility=volatility,
            best_bid=current_bid,
            best_ask=current_ask
        )

        # Generate intents for market making
        intents = []

        # Always place a bid (unless we're at max long inventory)
        if self.max_inventory is None or inventory < self.max_inventory:
            # Confidence based on how much better our bid is than current market
            if current_bid > 0 and optimal_bid > current_bid:
                bid_improvement_pct = (optimal_bid - current_bid) / current_bid
                confidence = min(bid_improvement_pct * 50.0, 1.0)
            else:
                confidence = 0.1

            confidence = max(confidence, 0.1)

            intents.append(Intent(
                listing=self.listing,
                side="B",
                confidence=confidence,
                price=optimal_bid
            ))

        # Always place an ask (unless we're at max short inventory)
        if self.max_inventory is None or inventory > -self.max_inventory:
            if current_ask > 0 and optimal_ask < current_ask:
                ask_improvement_pct = (current_ask - optimal_ask) / current_ask
                confidence = min(ask_improvement_pct * 50.0, 1.0)
            else:
                confidence = 0.1

            confidence = max(confidence, 0.1)

            intents.append(Intent(
                listing=self.listing,
                side="S",
                confidence=confidence,
                price=optimal_ask
            ))

        return intents
