from statsmodels.tsa.vector_ar.vecm import coint_johansen
from gnomepy.data.types import *
from gnomepy.research_old.trade_signal import *
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
        self.id = id(self)  # Add unique identifier

    def __hash__(self):
        return self.id

    def initialize_backtest(self) -> None:
        pass

    def process_event(self, listing_data: dict[Listing, pd.DataFrame]) -> list[TradeSignal | BasketTradeSignal]:
        pass

    def precompute_strategy_values(self, listing_data: dict[Listing, pd.DataFrame]) -> dict:
        """Precompute strategy-dependent values for vectorized backtesting.
        
        This method should be overridden by subclasses to precompute any values
        that would normally be computed during the iterative process.
        
        Args:
            listing_data: Dictionary mapping Listing objects to their full historical data DataFrames
            
        Returns:
            dict: Dictionary containing precomputed values needed for vectorized signal generation
        """
        return {}

    def generate_vectorized_signals(self, listing_data: dict[Listing, pd.DataFrame], 
                                  precomputed_values: dict) -> list[tuple[int, TradeSignal | BasketTradeSignal]]:
        """Generate signals using precomputed values for vectorized backtesting.
        
        This method should be overridden by subclasses to generate signals using
        the precomputed values from precompute_strategy_values.
        
        Args:
            listing_data: Dictionary mapping Listing objects to their full historical data DataFrames
            precomputed_values: Dictionary containing precomputed values from precompute_strategy_values
            
        Returns:
            list: List of tuples containing (timestamp, TradeSignal/BasketTradeSignal objects)
        """
        return []

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

        # Keep track of historical beta vectors
        self.beta_history = []
        self.beta_timestamps = []

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

    def __hash__(self):
        """Make strategy hashable by using its unique ID."""
        return self.id

    def initialize_backtest(self):
        # Strategy state variables
        self.beta_vec = None
        self.norm_beta_vec = None
        self.n_coints = None
        self.beta_history = []
        self.beta_timestamps = []
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

    def precompute_strategy_values(self, listing_data: dict[Listing, pd.DataFrame]) -> dict:
        """Precompute beta vectors and other strategy values for vectorized backtesting.
        
        Args:
            listing_data: Dictionary mapping Listing objects to their full historical data DataFrames
            
        Returns:
            dict: Dictionary containing precomputed beta vectors, timestamps, and other values
        """
        # Get the length of data and create timestamp array
        data_length = len(listing_data[self.listings[0]])
        timestamps = listing_data[self.listings[0]]['timestampEvent'].values
        
        # Initialize arrays to store precomputed values
        beta_vectors = np.full((data_length, len(self.listings)), np.nan)
        norm_beta_vectors = np.full((data_length, len(self.listings)), np.nan)
        n_coints_array = np.full(data_length, np.nan)
        z_scores = np.full(data_length, np.nan)
        
        # Create price matrix for all data
        price_matrix = np.column_stack([
            np.log(listing_data[listing]['bidPrice0'].values) 
            for listing in self.listings
        ])
        
        # Precompute beta vectors at beta refresh frequency intervals
        for i in range(self.beta_refresh_frequency, data_length, self.beta_refresh_frequency):
                
            # Calculate beta vectors for this window
            window_start = max(0, i - self.beta_refresh_frequency)
            coint_price_matrix = price_matrix[window_start:i+1]
            
            try:
                johansen_result = coint_johansen(coint_price_matrix, det_order=0, k_ar_diff=1)
                trace_stats = johansen_result.lr1
                cv = johansen_result.cvt[:, self.sig_idx]
                n_coints = np.sum(trace_stats > cv)

                if n_coints == 0:
                    if self.retest_cointegration:
                        # Set beta vectors to None for this period
                        end_idx = min(i + self.beta_refresh_frequency, data_length)
                        beta_vectors[i:end_idx] = np.nan
                        norm_beta_vectors[i:end_idx] = np.nan
                        n_coints_array[i:end_idx] = 0
                        continue
                    else:
                        n_coints = 1
                        beta_vec = johansen_result.evec[:, :n_coints]
                else:
                    # Use first beta vector
                    n_coints = 1
                    beta_vec = johansen_result.evec[:, :n_coints]
                
                norm_beta_vec = beta_vec / np.linalg.norm(beta_vec)
                
                # Store values for this period
                end_idx = min(i + self.beta_refresh_frequency, data_length)
                beta_vectors[i:end_idx] = beta_vec.flatten()
                norm_beta_vectors[i:end_idx] = norm_beta_vec.flatten()
                n_coints_array[i:end_idx] = n_coints
                
                # Store historical beta vector and timestamp
                self.beta_history.append(beta_vec.flatten())
                self.beta_timestamps.append(timestamps[i])
                
            except Exception as e:
                continue
        
        # Calculate z-scores for all timestamps where we have beta vectors
        for i in range(self.beta_refresh_frequency, data_length):
            if n_coints_array[i] > 0 and not np.any(np.isnan(norm_beta_vectors[i])):
                # Calculate spread using current beta vector and spread_window
                window_start = max(0, i - self.spread_window)
                window_spreads = price_matrix[window_start:i+1] @ beta_vectors[i].reshape(-1, 1)

                if len(window_spreads) > 1:
                    z_score = (window_spreads[-1][0] - window_spreads[:-1].mean()) / window_spreads[:-1].std()
                    z_scores[i] = z_score
        
        return {
            'beta_vectors': beta_vectors,
            'norm_beta_vectors': norm_beta_vectors,
            'n_coints_array': n_coints_array,
            'z_scores': z_scores,
            'timestamps': timestamps,
            'price_matrix': price_matrix,
            'beta_history': self.beta_history,
            'beta_timestamps': self.beta_timestamps
        }

    def generate_vectorized_signals(self, listing_data: dict[Listing, pd.DataFrame], 
                                  precomputed_values: dict) -> list[tuple[int, TradeSignal | BasketTradeSignal]]:
        """Generate signals using precomputed values for vectorized backtesting.
        
        Args:
            listing_data: Dictionary mapping Listing objects to their full historical data DataFrames
            precomputed_values: Dictionary containing precomputed values from precompute_strategy_values
            
        Returns:
            list: List of tuples containing (timestamp, TradeSignal/BasketTradeSignal objects)
        """
        signals = []
        data_length = len(listing_data[self.listings[0]])
        
        # Extract precomputed values
        norm_beta_vectors = precomputed_values['norm_beta_vectors']
        n_coints_array = precomputed_values['n_coints_array']
        z_scores = precomputed_values['z_scores']
        timestamps = precomputed_values['timestamps']
                
        # Generate signals for each timestamp
        for i in range(self.beta_refresh_frequency, data_length):
            # Skip if no cointegration, not at trade frequency, or at beta refresh point
            if n_coints_array[i] == 0 or np.isnan(z_scores[i]) or i % self.beta_refresh_frequency == 0:
                continue
            
            z_score = z_scores[i]
            norm_beta_vec = norm_beta_vectors[i]
            timestamp = timestamps[i]
            
            # Calculate confidence multiplier
            confidence_multiplier = min(abs(z_score / self.enter_zscore), 3.0)
            
            # Generate signals based on z-score thresholds
            if z_score < -self.enter_zscore:
                # Enter positive mean reversion
                signal_signals = [
                    TradeSignal(
                        listing=self.listings[j],
                        action=Action.BUY if norm_beta_vec[j] > 0 else Action.SELL,
                        confidence=confidence_multiplier
                    ) for j in range(len(self.listings))
                ]
                signals.append((timestamp, BasketTradeSignal(
                    signals=signal_signals,
                    proportions=norm_beta_vec,
                    strategy=self,
                    signal_type=SignalType.ENTER_POSITIVE_MEAN_REVERSION
                )))
                
            elif z_score > self.enter_zscore:
                # Enter negative mean reversion
                signal_signals = [
                    TradeSignal(
                        listing=self.listings[j],
                        action=Action.BUY if -norm_beta_vec[j] > 0 else Action.SELL,
                        confidence=1.0
                    ) for j in range(len(self.listings))
                ]
                signals.append((timestamp, BasketTradeSignal(
                    signals=signal_signals,
                    proportions=-norm_beta_vec,
                    strategy=self,
                    signal_type=SignalType.ENTER_NEGATIVE_MEAN_REVERSION
                )))
                
            elif (z_score < -self.enter_zscore - self.stop_loss_delta or z_score > -self.exit_zscore):
                # Exit positive reversion
                signal_signals = [
                    TradeSignal(
                        listing=self.listings[j],
                        action=Action.SELL if norm_beta_vec[j] > 0 else Action.BUY,
                        confidence=1.0
                    ) for j in range(len(self.listings))
                ]
                signals.append((timestamp, BasketTradeSignal(
                    signals=signal_signals,
                    proportions=norm_beta_vec,
                    strategy=self,
                    signal_type=SignalType.EXIT_POSITIVE_MEAN_REVERSION
                )))
                
            elif (z_score > self.enter_zscore + self.stop_loss_delta or z_score < self.exit_zscore):
                # Exit negative reversion
                signal_signals = [
                    TradeSignal(
                        listing=self.listings[j],
                        action=Action.SELL if -norm_beta_vec[j] > 0 else Action.BUY,
                        confidence=1.0
                    ) for j in range(len(self.listings))
                ]
                signals.append((timestamp, BasketTradeSignal(
                    signals=signal_signals,
                    proportions=-norm_beta_vec,
                    strategy=self,
                    signal_type=SignalType.EXIT_NEGATIVE_MEAN_REVERSION
                )))
        
        return signals
    
    def process_event(self, listing_data: dict[Listing, pd.DataFrame]) -> tuple[list[TradeSignal | BasketTradeSignal], float]:
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

            # Store historical beta vector and timestamp
            if self.beta_vec is not None:
                self.beta_history.append(self.beta_vec.flatten())
                self.beta_timestamps.append(idx)
            
            return [], time.time() - start_time

        # If not, then we can consider trading
        elif self.beta_vec is not None:

            # Create price matrix and calculate spread using spread_window
            coint_price_matrix = np.column_stack([np.log(listing_data[listing].loc[idx-self.spread_window:idx+1]['bidPrice0'].values) for listing in self.listings])

            # TODO:Implement LOB balance signal

            # Calculate past spreads and newest one using beta vector
            window_spreads = coint_price_matrix @ self.beta_vec

            # Calculate z score of newest time stamp
            z_score = (window_spreads[-1][0] - window_spreads[:-1].mean()) / window_spreads[:-1].std()

            # Turn z_score into confidence, cap at 3x
            confidence_multiplier = min(abs(z_score / self.enter_zscore), 3.0)

            # Positive mean reversion: b_l = [0.3, -0.4]  I'm waiting for a positive reversion. I've entered long on positive betas, so I sell positive betas on exit. I've entered short on negative betas, so I buy negative betas on exit.
            # Negative mean reversion: b_s = [-0.3, 0.4]  I'm waiting for a negative reversion. I've inverted my beta vector. I still enter long on positive betas, so I sell positive betas on exit. I still enter short on negative betas, so I buy negative betas on exit.
            # Enter positive mean reversion
            if z_score < -self.enter_zscore: #TODO: and (not self.use_lob or (self.use_lob and lob_signal)):
                signals = [TradeSignal(listing = self.listings[i], 
                                  action=Action.BUY if self.norm_beta_vec[i] > 0 else Action.SELL,
                                  confidence=confidence_multiplier) for i in range(len(self.listings))]

                return [BasketTradeSignal(signals=signals, proportions=self.norm_beta_vec, strategy=self, signal_type=SignalType.ENTER_POSITIVE_MEAN_REVERSION)], time.time() - start_time
            
            # Enter negative mean reversion
            elif z_score > self.enter_zscore:
                signals = [TradeSignal(listing = self.listings[i], 
                                  action=Action.BUY if -self.norm_beta_vec[i] > 0 else Action.SELL,
                                  confidence=1.0) for i in range(len(self.listings))]
                
                return [BasketTradeSignal(signals=signals, proportions=-self.norm_beta_vec, strategy=self, signal_type=SignalType.ENTER_NEGATIVE_MEAN_REVERSION)], time.time() - start_time

            # Exit positive reversion 
            elif (z_score < -self.enter_zscore - self.stop_loss_delta or z_score > -self.exit_zscore):
                signals = [TradeSignal(listing = self.listings[i], 
                                  action=Action.SELL if self.norm_beta_vec[i] > 0 else Action.BUY,
                                  confidence=1.0) for i in range(len(self.listings))]

                return [BasketTradeSignal(signals=signals, proportions=self.norm_beta_vec, strategy=self, signal_type=SignalType.EXIT_POSITIVE_MEAN_REVERSION)], time.time() - start_time

            # Exit negative reversion
            elif (z_score > self.enter_zscore + self.stop_loss_delta or z_score < self.exit_zscore):
                signals = [TradeSignal(listing = self.listings[i], 
                                  action=Action.SELL if -self.norm_beta_vec[i] > 0 else Action.BUY,
                                  confidence=1.0) for i in range(len(self.listings))]

                return [BasketTradeSignal(signals=signals, proportions=-self.norm_beta_vec, strategy=self, signal_type=SignalType.EXIT_NEGATIVE_MEAN_REVERSION)], time.time() - start_time

            # Add missing return for when no trading conditions are met
            return [], time.time() - start_time

        # We are not currently trading due to no more cointegration
        return [], time.time() - start_time
