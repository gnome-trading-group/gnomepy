#!/usr/bin/env python3
"""
Detailed debug script to compare iterative vs vectorized signal generation.
"""

import datetime
import numpy as np
import pandas as pd
from gnomepy.data.client import MarketDataClient
from gnomepy.data.types import Listing, SchemaType
from gnomepy.research_old.strategy import CointegrationStrategy
from gnomepy.research_old.backtest import Backtest, VectorizedBacktest

def detailed_comparison():
    """Detailed comparison between iterative and vectorized approaches."""
    
    # Initialize market data client
    client = MarketDataClient(bucket="gnome-market-data-prod", aws_profile_name="AWSAdministratorAccess-241533121172")
    
    # Define test parameters
    start_datetime = datetime.datetime(2025, 6, 18)
    end_datetime = datetime.datetime(2025, 6, 21)
    
    # Create test listings
    listings = [
        Listing(exchange_id=4, security_id=1),
        Listing(exchange_id=7, security_id=1),
    ]
    
    # Create strategy
    strategy = CointegrationStrategy(
        listings=listings,
        data_schema_type=SchemaType.MBP_10,
        trade_frequency=100,
        beta_refresh_frequency=1000,
        spread_window=100,
        enter_zscore=2.0,
        exit_zscore=0.3,
        stop_loss_delta=0.0,
        retest_cointegration=False,
        use_extends=True,
        use_lob=True,
        use_dynamic_sizing=True,
        significance_level=0.05
    )
    
    print("=== Fetching Data ===")
    # Fetch data once to use for both approaches
    iterative_backtest = Backtest(
        client=client,
        strategies=[strategy],
        start_datetime=start_datetime,
        end_datetime=end_datetime
    )
    
    # Get the data
    iterative_backtest._fetch_data()
    listing_data = iterative_backtest.listing_data
    
    print(f"Data shape: {len(listing_data[listings[0]])} rows")
    
    # Sample data at trade frequency
    sampled_data = {}
    for listing in listings:
        sampled_data[listing] = listing_data[listing].iloc[::strategy.trade_frequency].reset_index(drop=True)
    
    print(f"Sampled data shape: {len(sampled_data[listings[0]])} rows")
    
    print("\n=== Detailed Signal Comparison ===")
    
    # Initialize strategy for iterative approach
    strategy_iterative = CointegrationStrategy(
        listings=listings,
        data_schema_type=SchemaType.MBP_10,
        trade_frequency=100,
        beta_refresh_frequency=1000,
        spread_window=100,
        enter_zscore=2.0,
        exit_zscore=0.3,
        stop_loss_delta=0.0,
        retest_cointegration=False,
        use_extends=True,
        use_lob=True,
        use_dynamic_sizing=True,
        significance_level=0.05
    )
    strategy_iterative.initialize_backtest()
    
    # Initialize strategy for vectorized approach
    strategy_vectorized = CointegrationStrategy(
        listings=listings,
        data_schema_type=SchemaType.MBP_10,
        trade_frequency=100,
        beta_refresh_frequency=1000,
        spread_window=100,
        enter_zscore=2.0,
        exit_zscore=0.3,
        stop_loss_delta=0.0,
        retest_cointegration=False,
        use_extends=True,
        use_lob=True,
        use_dynamic_sizing=True,
        significance_level=0.05
    )
    strategy_vectorized.initialize_backtest()
    
    # Precompute values for vectorized approach
    precomputed_values = strategy_vectorized.precompute_strategy_values(sampled_data)
    
    print("\n=== Precomputed Values Analysis ===")
    print(f"Beta vectors shape: {precomputed_values['norm_beta_vectors'].shape}")
    print(f"Z-scores shape: {precomputed_values['z_scores'].shape}")
    print(f"Non-nan z-scores: {np.sum(~np.isnan(precomputed_values['z_scores']))}")
    print(f"Non-nan beta vectors: {np.sum(~np.isnan(precomputed_values['norm_beta_vectors'][:, 0]))}")
    
    # Track all signals for comparison
    iterative_signals = []
    vectorized_signals = []
    
    # Track position state for both approaches
    iterative_position = None
    vectorized_position = None
    
    # Compare first 50 timestamps in detail
    max_compare = min(50, len(sampled_data[listings[0]]))
    
    print(f"\n=== Comparing First {max_compare} Timestamps ===")
    
    for i in range(strategy.beta_refresh_frequency, max_compare):
        if i % strategy.trade_frequency != 0:
            continue
            
        print(f"\n--- Timestamp {i} ---")
        
        # Get data window for iterative
        window_start = max(0, i - strategy.max_lookback)
        window_end = i + 1
        iterative_data = {}
        for listing in listings:
            iterative_data[listing] = sampled_data[listing].iloc[window_start:window_end]
        
        # Generate iterative signal
        iterative_signal, _ = strategy_iterative.process_event(iterative_data)
        
        # Get vectorized values
        if not np.isnan(precomputed_values['z_scores'][i]):
            z_score_vec = precomputed_values['z_scores'][i]
            beta_vec = precomputed_values['norm_beta_vectors'][i]
            timestamp = precomputed_values['timestamps'][i]
            
            # Get iterative z-score and beta
            iterative_zscore = strategy_iterative._calculate_zscore(iterative_data) if strategy_iterative.norm_beta_vector is not None else None
            iterative_beta = strategy_iterative.norm_beta_vector
            
            print(f"  Timestamp: {timestamp}")
            print(f"  Iterative z-score: {iterative_zscore}")
            print(f"  Vectorized z-score: {z_score_vec}")
            print(f"  Z-score diff: {abs(iterative_zscore - z_score_vec) if iterative_zscore is not None else 'N/A'}")
            print(f"  Iterative beta: {iterative_beta}")
            print(f"  Vectorized beta: {beta_vec}")
            
            # Check if betas are different
            if iterative_beta is not None and not np.allclose(iterative_beta, beta_vec, rtol=1e-5):
                print(f"  BETA MISMATCH!")
                print(f"    Iterative: {iterative_beta}")
                print(f"    Vectorized: {beta_vec}")
                print(f"    Diff: {np.abs(iterative_beta - beta_vec)}")
            
            # Track iterative position state
            if iterative_signal:
                for sig in iterative_signal:
                    if sig.signal_type in ['enter_positive_mean_reversion', 'enter_negative_mean_reversion']:
                        iterative_position = sig.signal_type
                    elif sig.signal_type in ['exit_positive_mean_reversion', 'exit_negative_mean_reversion']:
                        iterative_position = sig.signal_type
            
            # Generate vectorized signal manually with same logic
            vectorized_signal_type = None
            if vectorized_position is None:
                if z_score_vec < -strategy.enter_zscore:
                    vectorized_signal_type = 'enter_positive_mean_reversion'
                    vectorized_position = 'enter_positive_mean_reversion'
                elif z_score_vec > strategy.enter_zscore:
                    vectorized_signal_type = 'enter_negative_mean_reversion'
                    vectorized_position = 'enter_negative_mean_reversion'
            elif vectorized_position == 'enter_positive_mean_reversion':
                if (z_score_vec < -strategy.enter_zscore - strategy.stop_loss_delta or z_score_vec > -strategy.exit_zscore):
                    vectorized_signal_type = 'exit_positive_mean_reversion'
                    vectorized_position = 'exit_positive_mean_reversion'
            elif vectorized_position == 'enter_negative_mean_reversion':
                if (z_score_vec > strategy.enter_zscore + strategy.stop_loss_delta or z_score_vec < strategy.exit_zscore):
                    vectorized_signal_type = 'exit_negative_mean_reversion'
                    vectorized_position = 'exit_negative_mean_reversion'
            elif vectorized_position in ['exit_positive_mean_reversion', 'exit_negative_mean_reversion']:
                vectorized_position = None  # Reset to allow new entries
            
            print(f"  Iterative position: {iterative_position}")
            print(f"  Vectorized position: {vectorized_position}")
            print(f"  Iterative signals: {[sig.signal_type for sig in iterative_signal]}")
            print(f"  Vectorized would generate: {vectorized_signal_type}")
            
            # Check for mismatches
            iterative_signal_types = [sig.signal_type for sig in iterative_signal]
            if vectorized_signal_type and vectorized_signal_type not in iterative_signal_types:
                print(f"  SIGNAL MISMATCH: Vectorized would generate {vectorized_signal_type} but iterative didn't")
            elif iterative_signal_types and not vectorized_signal_type:
                print(f"  SIGNAL MISMATCH: Iterative generated {iterative_signal_types} but vectorized wouldn't")
            
            iterative_signals.extend(iterative_signal)
            if vectorized_signal_type:
                vectorized_signals.append(vectorized_signal_type)
        else:
            print(f"  Vectorized: No valid z-score")
    
    print(f"\n=== Summary ===")
    print(f"Total iterative signals: {len(iterative_signals)}")
    print(f"Total vectorized signals: {len(vectorized_signals)}")
    
    # Compare signal types
    iterative_types = [sig.signal_type for sig in iterative_signals]
    print(f"\nIterative signal types: {iterative_types}")
    print(f"Vectorized signal types: {vectorized_signals}")
    
    # Check for differences in signal generation logic
    print(f"\n=== Key Differences Found ===")
    
    # 1. Position state tracking
    print("1. Position State Tracking:")
    print("   - Iterative: Tracks position state and validates signals against current position")
    print("   - Vectorized: Generates all signals without position state validation")
    
    # 2. Beta vector refresh timing
    print("\n2. Beta Vector Refresh Timing:")
    print("   - Iterative: Refreshes beta vectors every beta_refresh_frequency ticks")
    print("   - Vectorized: Precomputes beta vectors at fixed intervals")
    
    # 3. Z-score calculation window
    print("\n3. Z-score Calculation Window:")
    print("   - Iterative: Uses current data window for z-score calculation")
    print("   - Vectorized: Uses precomputed z-scores based on fixed windows")
    
    # 4. Signal validation
    print("\n4. Signal Validation:")
    print("   - Iterative: Validates signals against current position state")
    print("   - Vectorized: No position state validation during signal generation")

def compare_trade_logs(iterative_log, vectorized_log):
    print("\n=== Trade Log Comparison ===")
    # Sort by timestamp, listing, and action for deterministic comparison
    def prep(df):
        return df.sort_values(["timestamp", "listing", "action"]).reset_index(drop=True)
    iter_log = prep(iterative_log)
    vect_log = prep(vectorized_log)
    min_len = min(len(iter_log), len(vect_log))
    mismatches = 0
    for i in range(min_len):
        row_iter = iter_log.iloc[i]
        row_vect = vect_log.iloc[i]
        if not (
            row_iter["timestamp"] == row_vect["timestamp"] and
            row_iter["listing"] == row_vect["listing"] and
            row_iter["action"] == row_vect["action"] and
            np.isclose(row_iter["price"], row_vect["price"], rtol=1e-6) and
            np.isclose(row_iter["size"], row_vect["size"], rtol=1e-6)
        ):
            print(f"Mismatch at trade {i}:")
            print(f"  Iterative: {row_iter.to_dict()}")
            print(f"  Vectorized: {row_vect.to_dict()}")
            mismatches += 1
    if len(iter_log) != len(vect_log):
        print(f"Trade count mismatch: Iterative={len(iter_log)}, Vectorized={len(vect_log)}")
    if mismatches == 0 and len(iter_log) == len(vect_log):
        print("All trades match!")
    else:
        print(f"Total mismatches: {mismatches}")

def run_debug_comparison():
    # (Same setup as before)
    client = MarketDataClient(bucket="gnome-market-data-prod", aws_profile_name="AWSAdministratorAccess-241533121172")
    start_datetime = datetime.datetime(2025, 6, 18)
    end_datetime = datetime.datetime(2025, 6, 21)
    listings = [
        Listing(exchange_id=4, security_id=1),
        Listing(exchange_id=7, security_id=1),
    ]
    strategy = CointegrationStrategy(
        listings=listings,
        data_schema_type=SchemaType.MBP_10,
        trade_frequency=100,
        beta_refresh_frequency=1000,
        spread_window=100,
        enter_zscore=2.0,
        exit_zscore=0.3,
        stop_loss_delta=0.0,
        retest_cointegration=False,
        use_extends=True,
        use_lob=True,
        use_dynamic_sizing=True,
        significance_level=0.05
    )
    print("=== Running Iterative Backtest ===")
    iterative_backtest = Backtest(
        client=client,
        strategies=[strategy],
        start_datetime=start_datetime,
        end_datetime=end_datetime
    )
    iter_results = iterative_backtest.run()
    iter_trade_log = iter_results[1]
    print(f"Iterative trades: {len(iter_trade_log)}")
    print("Iterative trade log columns:", iter_trade_log.columns.tolist())
    print(iter_trade_log.head())
    print("=== Running Vectorized Backtest ===")
    vectorized_backtest = VectorizedBacktest(
        client=client,
        strategies=[strategy],
        start_datetime=start_datetime,
        end_datetime=end_datetime
    )
    vect_results = vectorized_backtest.run()
    vect_trade_log = vect_results[1]
    print(f"Vectorized trades: {len(vect_trade_log)}")
    print("Vectorized trade log columns:", vect_trade_log.columns.tolist())
    print(vect_trade_log.head())
    # Only keep relevant columns
    iter_log = iter_trade_log[["timestamp", "listing", "action", "price", "size"]].copy()
    vect_log = vect_trade_log[["timestamp", "listing", "action", "price", "size"]].copy()
    # Convert listing to string for comparison
    iter_log["listing"] = iter_log["listing"].astype(str)
    vect_log["listing"] = vect_log["listing"].astype(str)
    compare_trade_logs(iter_log, vect_log)

if __name__ == "__main__":
    run_debug_comparison() 