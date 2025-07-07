from statsmodels.tsa.vector_ar.vecm import coint_johansen
from gnomepy.data.types import *
from gnomepy.backtest.signal import *
import pandas as pd
import numpy as np
import time


# Global significance level mapping
SIGNIFICANCE_LEVEL_MAP = {0.01: 0, 0.05: 1, 0.10: 2}

class Strategy:

    def __init__(self, listings: list[Listing], data_schema_type: SchemaType, trade_frequency: int):
        self.listings = listings
        self.data_schema_type = data_schema_type
        self.trade_frequency = trade_frequency
        self.max_lookback = 0

    def initialize_backtest(self) -> None:
        pass

    def process_event(self, listing_data: dict[Listing, pd.DataFrame]) -> list[Signal | BasketSignal]:
        pass

class CointegrationStrategy(Strategy):

    def __init__(self, listings: list[Listing], data_schema_type: SchemaType = SchemaType.MBP_10,
                 trade_frequency: int = 1, beta_refresh_frequency: int = 1000,
                 spread_window: int = 100, enter_zscore: float = 2.0, exit_zscore: float = 0.3,
                 stop_loss_delta: float = 0.0, retest_cointegration: bool = False, use_extends: bool = True,
                 use_lob: bool = True, use_dynamic_sizing: bool = True, significance_level: float = 0.05):
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
        super().__init__(listings, data_schema_type, trade_frequency)

        self.beta_refresh_frequency = beta_refresh_frequency
        self.spread_window = spread_window
        self.enter_zscore = enter_zscore
        self.exit_zscore = exit_zscore
        self.stop_loss_delta = stop_loss_delta
        self.retest_cointegration = retest_cointegration
        self.use_extends = use_extends
        self.use_lob = use_lob
        self.use_dynamic_sizing = use_dynamic_sizing
        self.max_lookback = max(beta_refresh_frequency, spread_window)
        self.significance_level = significance_level
        self.sig_idx = SIGNIFICANCE_LEVEL_MAP[significance_level]

    def __str__(self):
        """Create string representation of strategy parameters."""
        return (
            f"CointegrationStrategy("
            f"listings={self.listings}, "
            f"data_schema_type={self.data_schema_type}, "
            f"trade_frequency={self.trade_frequency}, "
            f"beta_refresh_frequency={self.beta_refresh_frequency}, "
            f"spread_window={self.spread_window}, "
            f"enter_zscore={self.enter_zscore}, "
            f"exit_zscore={self.exit_zscore}, "
            f"stop_loss_delta={self.stop_loss_delta}, "
            f"retest_cointegration={self.retest_cointegration}, "
            f"use_extends={self.use_extends}, "
            f"use_lob={self.use_lob}, "
            f"use_dynamic_sizing={self.use_dynamic_sizing}, "
            f"significance_level={self.significance_level})"
        )

    def __eq__(self, other):
        """Define equality based on parameters."""
        if not isinstance(other, CointegrationStrategy):
            return False
        return (
            self.listings == other.listings and
            self.data_schema_type == other.data_schema_type and
            self.trade_frequency == other.trade_frequency and
            self.beta_refresh_frequency == other.beta_refresh_frequency and
            self.spread_window == other.spread_window and
            self.enter_zscore == other.enter_zscore and
            self.exit_zscore == other.exit_zscore and
            self.stop_loss_delta == other.stop_loss_delta and
            self.retest_cointegration == other.retest_cointegration and
            self.use_extends == other.use_extends and
            self.use_lob == other.use_lob and
            self.use_dynamic_sizing == other.use_dynamic_sizing and
            self.significance_level == other.significance_level
        )

    def initialize_backtest(self):
        # Strategy state variables
        self.beta_vec = None
        self.norm_beta_vec = None
        self.n_coints = None
        return

    def validate_signal(self, signal, strategy_position_state):
        """
        Validate if a signal is appropriate given current position state.
        
        Args:
            signal: BasketSignal with strategy and signal_type
            strategy_position_state: Current position state for this strategy
            
        Returns:
            bool: True if signal should be executed
        """
        current_position = strategy_position_state['position_type']
        
        if signal.signal_type == SignalType.ENTER_POSITIVE_MEAN_REVERSION:
            return current_position is None or current_position == SignalType.EXIT_POSITIVE_MEAN_REVERSION or current_position == SignalType.EXIT_NEGATIVE_MEAN_REVERSION  # Can enter if neutral or exited
        elif signal.signal_type == SignalType.ENTER_NEGATIVE_MEAN_REVERSION:
            return current_position is None or current_position == SignalType.EXIT_POSITIVE_MEAN_REVERSION or current_position == SignalType.EXIT_NEGATIVE_MEAN_REVERSION  # Can enter if neutral or exited
        elif signal.signal_type == SignalType.EXIT_POSITIVE_MEAN_REVERSION:
            return current_position == SignalType.ENTER_POSITIVE_MEAN_REVERSION
        elif signal.signal_type == SignalType.EXIT_NEGATIVE_MEAN_REVERSION:
            return current_position == SignalType.ENTER_NEGATIVE_MEAN_REVERSION
        
        return False
    
    def process_event(self, listing_data: dict[Listing, pd.DataFrame]) -> tuple[list[Signal | BasketSignal], float]:
        """Process market data event and generate trading signals.
        
        Returns:
            tuple containing:
                - list of Signal/BasketSignal objects
                - float latency in seconds
        """
        # Start latency calculation
        start_time = time.time()

        # Assert all dataframes have same length
        N = len(listing_data[self.listings[0]])
        idx = listing_data[self.listings[0]].index[-1]

        for listing in self.listings[1:]:
            assert len(listing_data[listing]) == N, f"DataFrame lengths don't match: {len(listing_data[listing])} != {N}"

        # Determine if there's enough data to run calculations 
        if N < self.max_lookback:
            return [], time.time() - start_time

        # First check if we need to update betas
        if idx % self.beta_refresh_frequency == 0:

            # Create price matrix and calculate beta vectors
            coint_price_matrix = np.column_stack([np.log(listing_data[listing].loc[idx-self.beta_refresh_frequency:idx+1]['bidPrice0'].values) for listing in self.listings])
            johansen_result = coint_johansen(coint_price_matrix, det_order=0, k_ar_diff=1)
            trace_stats = johansen_result.lr1
            cv = johansen_result.cvt[:, self.sig_idx]
            self.n_coints = np.sum(trace_stats > cv)

            # We tested and there is no more valid cointegration
            if self.n_coints == 0:

                # If this is True, then we will simply not trade during this beta refresh cycle
                if self.retest_cointegration:
                    self.beta_vec = None
                    self.norm_beta_vec = None

                # If we want to trade regardless 
                else:
                    self.n_coints = 1
                    self.beta_vec = johansen_result.evec[:, :self.n_coints]
                    self.norm_beta_vec = self.beta_vec / np.linalg.norm(self.beta_vec)

            else:
                ## OPTIONAL: RIGHT NOW ITS EASIER TO JUST TRADE THE FIRST BETA VECTOR
                self.n_coints = 1
                ## OPTIONAL

                self.beta_vec = johansen_result.evec[:, :self.n_coints]
                self.norm_beta_vec = self.beta_vec / np.linalg.norm(self.beta_vec)
            
            return [], time.time() - start_time

        # If not, then we can consider trading
        elif self.beta_vec is not None:

            # Create price matrix and calculate spread
            coint_price_matrix = np.column_stack([np.log(listing_data[listing].loc[idx-self.beta_refresh_frequency:idx+1]['bidPrice0'].values) for listing in self.listings])

            # TODO:Implement LOB balance signal

            # Caclulate past spreads and newest one
            window_spreads = coint_price_matrix @ self.beta_vec

            # Calculate z score of newest time stamp
            z_score = (window_spreads[-1][0] - window_spreads[:-1].mean()) / window_spreads[:-1].std()

            # Turn z_score into confidence, cap at 3x
            confidence_multiplier = min(abs(z_score / self.enter_zscore), 3.0)

            # Positive mean reversion: b_l = [0.3, -0.4]  I'm waiting for a positive reversion. I've entered long on positive betas, so I sell positive betas on exit. I've entered short on negative betas, so I buy negative betas on exit.
            # Negative mean reversion: b_s = [-0.3, 0.4]  I'm waiting for a negative reversion. I've inverted my beta vector. I still enter long on positive betas, so I sell positive betas on exit. I still enter short on negative betas, so I buy negative betas on exit.
            # Enter positive mean reversion
            if z_score < -self.enter_zscore: #TODO: and (not self.use_lob or (self.use_lob and lob_signal)):
                signals = [Signal(listing = self.listings[i], 
                                  action=Action.BUY if self.norm_beta_vec[i] > 0 else Action.SELL,
                                  confidence=confidence_multiplier) for i in range(len(self.listings))]

                return [BasketSignal(signals=signals, proportions=self.norm_beta_vec, strategy=self, signal_type=SignalType.ENTER_POSITIVE_MEAN_REVERSION)], time.time() - start_time
            
            # Enter negative mean reversion
            elif z_score > self.enter_zscore:
                signals = [Signal(listing = self.listings[i], 
                                  action=Action.BUY if -self.norm_beta_vec[i] > 0 else Action.SELL,
                                  confidence=1.0) for i in range(len(self.listings))]
                
                return [BasketSignal(signals=signals, proportions=-self.norm_beta_vec, strategy=self, signal_type=SignalType.ENTER_NEGATIVE_MEAN_REVERSION)], time.time() - start_time

            # Exit positive reversion 
            elif (z_score < -self.enter_zscore - self.stop_loss_delta or z_score > -self.exit_zscore):
                signals = [Signal(listing = self.listings[i], 
                                  action=Action.SELL if self.norm_beta_vec[i] > 0 else Action.BUY,
                                  confidence=1.0) for i in range(len(self.listings))]

                return [BasketSignal(signals=signals, proportions=self.norm_beta_vec, strategy=self, signal_type=SignalType.EXIT_POSITIVE_MEAN_REVERSION)], time.time() - start_time

            # Exit negative reversion
            elif (z_score > self.enter_zscore + self.stop_loss_delta or z_score < self.exit_zscore):
                signals = [Signal(listing = self.listings[i], 
                                  action=Action.SELL if -self.norm_beta_vec[i] > 0 else Action.BUY,
                                  confidence=1.0) for i in range(len(self.listings))]

                return [BasketSignal(signals=signals, proportions=-self.norm_beta_vec, strategy=self, signal_type=SignalType.EXIT_NEGATIVE_MEAN_REVERSION)], time.time() - start_time

            # Add missing return for when no trading conditions are met
            return [], time.time() - start_time

        # We are not currently trading due to no more cointegration
        return [], time.time() - start_time
