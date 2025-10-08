from abc import ABC, abstractmethod
from gnomepy.data.types import SchemaBase, SchemaType
from gnomepy.registry.types import Listing
from gnomepy.research.types import Intent, BasketIntent
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import numpy as np
import pandas as pd

# Global significance level mapping
SIGNIFICANCE_LEVEL_MAP = {0.01: 0, 0.05: 1, 0.10: 2}

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


class CointegrationSignal(PositionAwareSignal):

    def __init__(self, listings: list[Listing], data_schema_type: SchemaType = SchemaType.MBP_10,
                 trade_frequency: int = 1, beta_refresh_frequency: int = 1000,
                 spread_window: int = 100, enter_zscore: float = 2.0, exit_zscore: float = 0.3,
                 stop_loss_delta: float = 0.0, retest_cointegration: bool = True, use_extends: bool = False,
                 use_lob: bool = False, use_dynamic_sizing: bool = True, significance_level: float = 0.05):
        """Initialize a cointegration trading strategy.
        
        Parameters
        ----------
        listings : list[Listing]
            List of listings to trade as a cointegrated basket
        data_schema_type : SchemaType, default SchemaType.MBP_10
            Type of market data schema to use
        trade_frequency : int, default 1
            How frequently to check for trading signals
        beta_refresh_frequency : int, default 1000
            How frequently to recalculate cointegration betas
        spread_window : int, default 100
            Rolling window size for calculating spread statistics
        enter_zscore : float, default 2.0
            Z-score threshold to enter positions
        exit_zscore : float, default 0.3
            Z-score threshold to exit positions
        stop_loss_delta : float, default 0
            Stop loss threshold, 0 means no stop loss
        retest_cointegration : bool, default False
            Whether to retest cointegration at beta refresh frequency level
        use_extends : bool, default True
            Whether to allow extending positions
        use_lob : bool, default True
            Whether to use limit order book data
        use_dynamic_sizing : bool, default False
            Whether to use dynamic position sizing
        significance_level : float, default 0.05
            Significance level for cointegration testing (0.01, 0.05, or 0.10)
        """
        # Call parent constructor to set up _id
        super().__init__()

        # Get all the signal settings
        self.listings = listings
        self.data_schema_type = data_schema_type
        self.trade_frequency = trade_frequency
        self.beta_refresh_frequency = beta_refresh_frequency
        self.spread_window = spread_window
        self.enter_zscore = enter_zscore
        self.exit_zscore = exit_zscore
        self.stop_loss_delta = stop_loss_delta
        self.retest_cointegration = retest_cointegration
        self.use_extends = use_extends
        self.use_lob = use_lob
        self.use_dynamic_sizing = use_dynamic_sizing
        self.significance_level = significance_level
        self.max_lookback = max(beta_refresh_frequency, spread_window)
        self.sig_idx = SIGNIFICANCE_LEVEL_MAP[self.significance_level]

        # Prepare the signal to start generating intents
        self.initialize_signal()

    def initialize_signal(self):
        self.beta_vec = None
        self.norm_beta_vec = None
        self.n_coints = None
        self.beta_history = []
        self.beta_timestamps = []
        self.elapsed_ticks = {listing.listing_id: 0 for listing in self.listings}
        print(f"Initialized CointegrationSignal with {len(self.listings)} listings")
        return

    def refresh_beta_vec(self, data: dict[int, dict[str, np.ndarray]], positions: dict[int, float] = None):
        print(f"Refreshing beta vectors for CointegrationSignal...")

        # Before refreshing beta vectors, check if we need to flatten positions
        if positions is not None:
            has_positions = any(abs(positions.get(listing.listing_id, 0)) > 1e-6 for listing in self.listings)
            
            if has_positions:
                print(f"Beta refresh triggered - flattening positions before updating beta vectors")
                # Create flatten intents for all listings with positions
                flatten_intents = []
                for listing in self.listings:
                    current_position = positions.get(listing.listing_id, 0)
                    if abs(current_position) > 1e-6:
                        flatten_intents.append(Intent(
                            listing=listing,
                            side="S" if current_position > 0 else "B",
                            confidence=1.0,
                            flatten=True
                        ))
                
                if flatten_intents:
                    return [BasketIntent(
                        intents=flatten_intents,
                        proportions=[1.0] * len(flatten_intents)
                    )]

        # Create price matrix and calculate beta vectors - now using numpy arrays directly
        coint_price_matrix = np.column_stack([
            data[listing.listing_id]['bidPrice0'][-self.beta_refresh_frequency:] 
            for listing in self.listings
        ])
        johansen_result = coint_johansen(coint_price_matrix, det_order=0, k_ar_diff=1)
        self.n_coints = np.sum(johansen_result.lr1 > johansen_result.cvt[:, self.sig_idx])
        
        # If no cointegration found and we want to retest, skip trading
        if self.n_coints == 0 and self.retest_cointegration:
            print(f"No cointegration found (retest enabled) - beta vectors set to None")
            self.beta_vec = None
            self.norm_beta_vec = None
        else:
            # Use first eigenvector whether cointegration exists or not
            self.n_coints = 1
            self.beta_vec = johansen_result.evec[:, :self.n_coints]
            self.norm_beta_vec = self.beta_vec / np.linalg.norm(self.beta_vec)
            print(f"Beta vectors updated: {self.norm_beta_vec.flatten()}")

        # Store historical beta vector and timestamp
        if self.beta_vec is not None:
            self.beta_history.append(self.beta_vec.flatten())
            self.beta_timestamps.append(self.elapsed_ticks)

        return None
    
    def generate_intents(self, data_df: dict[int, dict[str, np.ndarray]], positions: dict[int, float] = None) -> list[BasketIntent]:
        """Generate trading intents based on current market data and beta vectors.
        
        Args:
            data_df: Dictionary mapping listing_id to their historical data (numpy arrays)
            positions: Dictionary mapping listing_id to their current positions
            
        Returns:
            list: List of BasketIntent objects
        """
        # Create price matrix and calculate spread using spread_window - now using numpy arrays directly
        coint_price_matrix = np.column_stack([
            data_df[listing.listing_id]['bidPrice0'][-self.spread_window:]
            for listing in self.listings
        ])

        # Calculate past spreads and newest one using beta vector
        window_spreads = coint_price_matrix @ self.beta_vec

        # Calculate z score of newest time stamp
        z_score = (window_spreads[-1][0] - window_spreads[:-1].mean()) / window_spreads[:-1].std()

        # Turn z_score into confidence, cap at 3x
        confidence_multiplier = min(abs(z_score / self.enter_zscore), 3.0)

        # Entrance condition: positions are near zero (within 1e-6)
        entrance_condition = all(abs(positions[listing.listing_id]) < 1e-6 for listing in self.listings)

        # Exit condition for positive mean reversion: positions have same sign as beta vectors AND all positions are nonzero
        positive_exit_condition = all(
            abs(positions[self.listings[i].listing_id]) > 1e-6 and
            ((positions[self.listings[i].listing_id] > 0 and self.norm_beta_vec[i] > 0) or
             (positions[self.listings[i].listing_id] < 0 and self.norm_beta_vec[i] < 0))
            for i in range(len(self.listings))
        )

        # Exit condition for negative mean reversion: positions have same sign as inverse beta vectors AND all positions are nonzero
        negative_exit_condition = all(
            abs(positions[self.listings[i].listing_id]) > 1e-6 and
            ((positions[self.listings[i].listing_id] > 0 and -self.norm_beta_vec[i] > 0) or
             (positions[self.listings[i].listing_id] < 0 and -self.norm_beta_vec[i] < 0))
            for i in range(len(self.listings))
        )

        # Check if we can extend positions when use_extends is True
        # Positions are aligned with beta vectors for positive mean reversion
        can_extend_positive = (self.use_extends and 
                             all(abs(positions[self.listings[i].listing_id]) > 1e-6 and
                                 ((positions[self.listings[i].listing_id] > 0 and self.norm_beta_vec[i] > 0) or
                                  (positions[self.listings[i].listing_id] < 0 and self.norm_beta_vec[i] < 0))
                                 for i in range(len(self.listings))))

        # Positions are aligned with inverse beta vectors for negative mean reversion
        can_extend_negative = (self.use_extends and 
                             all(abs(positions[self.listings[i].listing_id]) > 1e-6 and
                                 ((positions[self.listings[i].listing_id] > 0 and -self.norm_beta_vec[i] > 0) or
                                  (positions[self.listings[i].listing_id] < 0 and -self.norm_beta_vec[i] < 0))
                                 for i in range(len(self.listings))))

        # Enter positive mean reversion - only if entrance condition is met OR can extend
        if z_score < -self.enter_zscore and (entrance_condition or can_extend_positive):
            print(f"Triggered positive mean reversion entry (z_score: {z_score:.3f})")
            intents = [Intent(
                listing=self.listings[i],
                side="B" if self.norm_beta_vec[i] > 0 else "S",
                confidence=confidence_multiplier
            ) for i in range(len(self.listings))]
            
            return [BasketIntent(
                intents=intents,
                proportions=self.norm_beta_vec.flatten().tolist()
            )]
        
        # Enter negative mean reversion - only if entrance condition is met OR can extend
        elif z_score > self.enter_zscore and (entrance_condition or can_extend_negative):
            print(f"Triggered negative mean reversion entry (z_score: {z_score:.3f})")
            intents = [Intent(
                listing=self.listings[i],
                side="B" if -self.norm_beta_vec[i] > 0 else "S",
                confidence=1.0
            ) for i in range(len(self.listings))]
            
            return [BasketIntent(
                intents=intents,
                proportions=(-self.norm_beta_vec).flatten().tolist()
            )]

        # Exit positive reversion - only if positive exit condition is met
        elif (z_score < -self.enter_zscore - self.stop_loss_delta or z_score > -self.exit_zscore) and positive_exit_condition:
            print(f"Triggered positive reversion exit success (z_score: {z_score:.3f})")
            intents = [Intent(
                listing=self.listings[i],
                side="S" if self.norm_beta_vec[i] > 0 else "B",
                confidence=1.0,
                flatten=True
            ) for i in range(len(self.listings))]
            
            return [BasketIntent(
                intents=intents,
                proportions=self.norm_beta_vec.flatten().tolist()
            )]

        # Exit negative reversion - only if negative exit condition is met
        elif (z_score > self.enter_zscore + self.stop_loss_delta or z_score < self.exit_zscore) and negative_exit_condition:
            print(f"Triggered negative reversion exit success (z_score: {z_score:.3f})")
            intents = [Intent(
                listing=self.listings[i],
                side="S" if -self.norm_beta_vec[i] > 0 else "B",
                confidence=1.0,
                flatten=True
            ) for i in range(len(self.listings))]
            
            return [BasketIntent(
                intents=intents,
                proportions=(-self.norm_beta_vec).flatten().tolist()
            )]

        # No trading signal generated
        return []

    def process_new_tick(self, data: dict[int, dict[str, np.ndarray]], ticker_listing_id: int, positions: dict[int, float] = None) -> list[BasketIntent]:
        """Process market data event and generate trading signals.
        
        Args:
            data: Dictionary mapping listing_id to their historical data (numpy arrays)
            ticker_listing_id: The specific listing_id that received new data
            positions: Dictionary mapping listing_id to current position size
        
        Returns:
            list of BasketIntent objects
        """
        # Increment elapsed ticks for the specific listing that received new data
        if ticker_listing_id in self.elapsed_ticks:
            self.elapsed_ticks[ticker_listing_id] += 1

        # Determine if there's enough data to run calculations 
        # Check that each listing has enough data considering trade_frequency
        enough_data = all(
            (self.elapsed_ticks[listing.listing_id] // self.trade_frequency) >= self.max_lookback 
            for listing in self.listings
        )
        
        if not enough_data:
            return []

        # First check if we need to update betas
        # Use the minimum elapsed ticks across all listings to determine beta refresh timing
        min_elapsed_ticks = min(self.elapsed_ticks[listing.listing_id] for listing in self.listings)
        
        if min_elapsed_ticks % self.beta_refresh_frequency == 0:

            # Refresh beta vectors using cointegration testing (may return flatten intents)
            result = self.refresh_beta_vec(data, positions)
            
            if result is not None:
                # We got flatten intents, return them
                return result
            
            return []

        # If not, then we can consider trading
        elif self.beta_vec is not None:

            # Generate intents
            basket_intents = self.generate_intents(data, positions)

            return basket_intents


        # We are not currently trading due to no cointegration
        return []
