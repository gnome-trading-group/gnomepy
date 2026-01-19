from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from gnomepy.data.types import SchemaBase, SchemaType
from gnomepy.registry.types import Listing
from gnomepy.research.types import Intent, BasketIntent
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from gnomepy.backtest.recorder import ModelValueRecorder


class Signal(ABC):

    def __init__(self):
        self._id = id(self)

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        return isinstance(other, Signal) and self._id == other._id

    @abstractmethod
    def process_new_tick(self, data: SchemaBase) -> list[Intent]:
        """
        Process market data and return intents.

        Returns:
            list of Intent objects
        """
        raise NotImplementedError


class PositionAwareSignal(Signal):

    @abstractmethod
    def process_new_tick(self, data: SchemaBase, positions: dict[int, float] = None) -> list[Intent]:
        """
        Process market data and return intents, considering current positions.

        Args:
            data: Market data to process
            positions: Dictionary mapping listing_id to current position size

        Returns:
            list of Intent objects
        """
        raise NotImplementedError


class MarketMakingSignal(PositionAwareSignal):
    """Abstract base class for market making signals.

    This class provides common functionality for market making strategies,
    including volatility calculation, intent generation, and inventory management.
    Subclasses must implement calculate_optimal_prices to define their specific
    pricing model.
    """

    def __init__(
            self,
            listing: Listing,
            data_schema_type: SchemaType = SchemaType.MBP_10,
            trade_frequency: int = 1,
            volatility_window: int = 100,
            volatility_span: float = None,
            max_inventory: float = None,
            liquidation_threshold: float = 0.8,
            use_market_orders_for_liquidation: bool = True,
            model_value_recorder: 'ModelValueRecorder | None' = None
    ):
        """Initialize a market making signal.

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
        max_inventory : float, optional
            Maximum absolute inventory size before stopping market making
        liquidation_threshold : float, default 0.8
            Fraction of max_inventory at which to start aggressive liquidation (0.0-1.0)
        use_market_orders_for_liquidation : bool, default True
            If True, use market orders when at max inventory for immediate liquidation
        model_value_recorder : ModelValueRecorder, optional
            Recorder for model values. If None, a default recorder will be created.
        """
        super().__init__()

        self.listing = listing
        self.listings = [listing]  # For compatibility with OMS
        self.data_schema_type = data_schema_type
        self.trade_frequency = trade_frequency
        self.volatility_window = volatility_window  # Kept for backward compatibility
        # Set volatility_span: use provided value or default to volatility_window
        self.volatility_span = volatility_span if volatility_span is not None else float(volatility_window)
        self.max_inventory = max_inventory
        self.liquidation_threshold = liquidation_threshold
        self.use_market_orders_for_liquidation = use_market_orders_for_liquidation
        self.max_lookback = volatility_window  # Still use window for lookback

        # Initialize model value recorder
        if model_value_recorder is None:
            from gnomepy.backtest.recorder import ModelValueRecorder
            # Create a default recorder with just this listing
            self.model_value_recorder = ModelValueRecorder([listing.listing_id])
        else:
            self.model_value_recorder = model_value_recorder

        # Initialize state (elapsed_ticks only used for trade_frequency)
        self.elapsed_ticks = {listing.listing_id: 0}

    @abstractmethod
    def calculate_optimal_prices(
            self,
            mid_price: float,
            inventory: float,
            volatility: float,
    ) -> tuple[float, float]:
        """Calculate optimal bid and ask prices using the market making model.

        This method must be implemented by subclasses to define their specific
        pricing model (e.g., Avellaneda-Stoikov, GLFT, etc.).

        Args:
            mid_price: Current mid price
            inventory: Current inventory (position size)
            volatility: Volatility (sigma) per tick

        Returns:
            Tuple of (optimal_bid, optimal_ask)
        """
        raise NotImplementedError

    def calculate_volatility(self, mid_prices: np.ndarray) -> float:
        """Calculate exponentially weighted volatility per tick from mid prices.

        Uses exponentially weighted moving average (EWM) to give more weight
        to recent price movements while still considering historical data.

        Args:
            mid_prices: Array of mid prices

        Returns:
            Exponentially weighted volatility per tick (sigma)
        """
        if len(mid_prices) < 2:
            return 0.0

        # Calculate log returns
        log_returns = np.diff(np.log(mid_prices))

        if len(log_returns) == 0:
            return 0.0

        # Calculate exponentially weighted standard deviation
        # Convert to pandas Series for ewm() method
        returns_series = pd.Series(log_returns)

        # Use exponentially weighted moving standard deviation
        # span parameter controls the decay: alpha = 2/(span+1)
        # Higher span = more weight to older data (smoother)
        ewm_std = returns_series.ewm(span=self.volatility_span, adjust=False).std()

        # Return the most recent (last) volatility value
        # This gives the current volatility estimate with exponential weighting
        volatility_per_tick = ewm_std.iloc[-1]

        # Handle NaN (can occur if not enough data)
        if np.isnan(volatility_per_tick):
            # Fallback to simple std if EWM hasn't converged yet
            volatility_per_tick = np.std(log_returns)

        return float(volatility_per_tick)

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

        # For EWM volatility, we need at least 2 prices (for 1 return)
        # But use more data for better EWM convergence
        min_data_points = max(2, int(self.volatility_span * 0.5))  # At least half the span
        if len(bid_prices) < min_data_points or len(ask_prices) < min_data_points:
            return []

        # Get current mid price
        current_bid = bid_prices[-1]
        current_ask = ask_prices[-1]
        current_mid = (current_bid + current_ask) / 2.0

        # For EWM, use all available data (up to max_lookback) since EWM naturally weights recent data more
        lookback_size = min(len(bid_prices), self.max_lookback)
        recent_bids = bid_prices[-lookback_size:]
        recent_asks = ask_prices[-lookback_size:]
        recent_mids = (recent_bids + recent_asks) / 2.0

        # Calculate exponentially weighted volatility per tick
        volatility = self.calculate_volatility(recent_mids)

        # Avoid issues with zero or very small volatility
        if volatility < 1e-8:
            volatility = 1e-8

        # Get current inventory
        inventory = positions.get(listing_id, 0.0) if positions else 0.0

        # Check max inventory constraint - aggressive liquidation
        if self.max_inventory is not None and abs(inventory) >= self.max_inventory:
            # If at max inventory, liquidate immediately
            if abs(inventory) > 1e-6:
                # Use market orders for immediate liquidation if enabled
                if self.use_market_orders_for_liquidation:
                    return [Intent(
                        listing=self.listing,
                        side="S" if inventory > 0 else "B",
                        confidence=1.0,
                        flatten=True,
                        price=None  # None means market order
                    )]
                else:
                    # Use aggressive limit order - cross the spread to ensure fill
                    if inventory > 0:
                        liquidation_price = current_bid * 0.999
                    else:
                        liquidation_price = current_ask * 1.001

                    return [Intent(
                        listing=self.listing,
                        side="S" if inventory > 0 else "B",
                        confidence=1.0,
                        flatten=True,
                        price=liquidation_price
                    )]
            return []

        # Check if we're near max inventory - start aggressive liquidation
        liquidation_threshold_inventory = None
        if self.max_inventory is not None:
            liquidation_threshold_inventory = self.max_inventory * self.liquidation_threshold

        near_max_inventory = (
                liquidation_threshold_inventory is not None and
                abs(inventory) >= liquidation_threshold_inventory
        )

        # Calculate optimal bid and ask prices using the subclass implementation
        optimal_bid, optimal_ask = self.calculate_optimal_prices(
            current_mid, inventory, volatility
        )

        # Log model values if timestamp is provided
        if timestamp is not None:
            self.model_value_recorder.log_model_value(
                listing_id=listing_id,
                timestamp=timestamp,
                mid_price=current_mid,
                inventory=inventory,
                volatility=volatility,
                optimal_bid=optimal_bid,
                optimal_ask=optimal_ask,
                reservation_price=(optimal_bid + optimal_ask) / 2.0,  # Approximate reservation price
                spread=optimal_ask - optimal_bid
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

            if near_max_inventory and inventory < 0:
                confidence = min(confidence * 1.5, 1.0)

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

            if near_max_inventory and inventory > 0:
                confidence = min(confidence * 1.5, 1.0)

            intents.append(Intent(
                listing=self.listing,
                side="S",
                confidence=confidence,
                price=optimal_ask
            ))

        return intents

    def process_new_tick(
            self,
            data: dict[int, dict[str, np.ndarray]],
            ticker_listing_id: int,
            positions: dict[int, float] = None,
            timestamp: int = None
    ) -> list[Intent]:
        """Process market data event and generate trading signals.

        Args:
            data: Dictionary mapping listing_id to their historical data (numpy arrays)
            ticker_listing_id: The specific listing_id that received new data
            positions: Dictionary mapping listing_id to current position size
            timestamp: Optional timestamp for logging model values

        Returns:
            list of Intent objects
        """
        # Only process ticks for our listing
        if ticker_listing_id != self.listing.listing_id:
            return []

        # Check if we have enough data
        if ticker_listing_id not in data:
            return []

        listing_data = data[ticker_listing_id]

        # Check if we have required price data
        if 'bidPrice0' not in listing_data or 'askPrice0' not in listing_data:
            return []

        # Check if we have enough data for volatility calculation
        bid_prices = listing_data['bidPrice0']
        ask_prices = listing_data['askPrice0']

        # For EWM volatility, we need at least 2 prices (for 1 return)
        min_data_points = max(2, int(self.volatility_span * 0.5))
        if len(bid_prices) < min_data_points or len(ask_prices) < min_data_points:
            return []

        # Increment elapsed ticks (only used for trade_frequency)
        if ticker_listing_id not in self.elapsed_ticks:
            self.elapsed_ticks[ticker_listing_id] = 0
        self.elapsed_ticks[ticker_listing_id] += 1

        # Check trade frequency
        if self.elapsed_ticks[ticker_listing_id] % self.trade_frequency != 0:
            return []

        # Generate intents with optional timestamp
        return self.generate_intents(data, positions, timestamp=timestamp)
