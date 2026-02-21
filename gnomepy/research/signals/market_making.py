from abc import abstractmethod
from typing import TYPE_CHECKING
from gnomepy.data.types import SchemaType
from gnomepy.registry.types import Listing
from gnomepy.research.types import Intent
from gnomepy.research.signals import PositionAwareSignal
import numpy as np

if TYPE_CHECKING:
    from gnomepy.backtest.recorder import MarketRecorder


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
            max_lookback: int = 2,
            max_inventory: float | None = None,
            liquidation_threshold: float = 0.8,
            use_market_orders_for_liquidation: bool = True,
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
        max_lookback : int, default 2
            Maximum number of ticks to look back for volatility calculation.
        liquidation_threshold : float, default 0.8
            Fraction of max_inventory at which to start aggressive liquidation (0.0-1.0)
        use_market_orders_for_liquidation : bool, default True
            If True, use market orders when at max inventory for immediate liquidation
        """
        super().__init__()

        self.listing = listing
        self.listings = [listing]  # For compatibility with OMS
        self.data_schema_type = data_schema_type
        self.trade_frequency = trade_frequency
        self.max_lookback = max_lookback
        self.max_inventory = max_inventory
        self.liquidation_threshold = liquidation_threshold
        self.use_market_orders_for_liquidation = use_market_orders_for_liquidation

        # Initialize state (elapsed_ticks only used for trade_frequency)
        self.elapsed_ticks = {listing.listing_id: 0}

    @abstractmethod
    def generate_intents(
            self,
            data: dict[int, dict[str, np.ndarray]],
            positions: dict[int, float] = None,
            timestamp: int = None
    ) -> list[Intent]:
        """Generate trading intents based on optimal prices from calculate_optimal_prices."""
        raise NotImplementedError

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
        min_data_points = max(2, self.max_lookback)
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
