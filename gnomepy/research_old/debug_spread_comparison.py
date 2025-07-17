#!/usr/bin/env python3
"""
Comprehensive analysis script to compare iterative vs vectorized backtesting.
This script systematically examines each component to identify differences.
"""

import datetime
import numpy as np
import pandas as pd
from gnomepy.data.client import MarketDataClient
from gnomepy.data.types import Listing, SchemaType
from gnomepy.research_old.strategy import CointegrationStrategy
from gnomepy.research_old.backtest import Backtest, VectorizedBacktest

def comprehensive_analysis():
    """Comprehensive analysis of iterative vs vectorized backtesting differences."""
    
    print("=== COMPREHENSIVE ANALYSIS: ITERATIVE VS VECTORIZED ===")
    
    # Initialize market data client
    client = MarketDataClient(bucket="gnome-market-data-prod", aws_profile_name="AWSAdministratorAccess-241533121172")
    
    # Define test parameters
    start_datetime = datetime.datetime(2025, 6, 18)
    end_datetime = datetime.datetime(2025, 6, 21)
    
    # Create test listings
    listings = [
        Listing(exchange_id=4, security_id=1),
        Listing(exchange_id=1, security_id=1),
    ]
    
    # Create strategy with smaller parameters for easier debugging
    strategy = CointegrationStrategy(
        listings=listings,
        data_schema_type=SchemaType.MBP_10,
        trade_frequency=100,  # Trade every 100 ticks
        beta_refresh_frequency=1000,  # Refresh betas every 1000 ticks
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
    
    print(f"Strategy parameters:")
    print(f"  Trade frequency: {strategy.trade_frequency}")
    print(f"  Beta refresh frequency: {strategy.beta_refresh_frequency}")
    print(f"  Spread window: {strategy.spread_window}")
    print(f"  Enter z-score: {strategy.enter_zscore}")
    print(f"  Exit z-score: {strategy.exit_zscore}")
    
    # ===================================================================
    # STEP 1: COMPARE DATASETS
    # ===================================================================
    print("\n" + "="*60)
    print("STEP 1: DATASET COMPARISON")
    print("="*60)
    
    # Run both backtests to get their data
    print("Running iterative research_old...")
    iterative_backtest = Backtest(
        client=client,
        strategies=[strategy],
        start_datetime=start_datetime,
        end_datetime=end_datetime
    )
    
    # Get iterative data
    iterative_data = iterative_backtest.listing_data
    print(f"Iterative data length: {len(iterative_data[listings[0]])}")
    
    # Create fresh strategy for vectorized
    strategy_vec = CointegrationStrategy(
        listings=listings,
        data_schema_type=SchemaType.MBP_10,
        trade_frequency=100,  # Trade every 100 ticks
        beta_refresh_frequency=1000,  # Refresh betas every 1000 ticks
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
    
    print("Running vectorized research_old...")
    vectorized_backtest = VectorizedBacktest(
        client=client,
        strategies=[strategy_vec],
        start_datetime=start_datetime,
        end_datetime=end_datetime
    )
    
    # Get vectorized data
    vectorized_data = vectorized_backtest.listing_data
    print(f"Vectorized data length: {len(vectorized_data[listings[0]])}")
    
    # Compare data lengths
    if len(iterative_data[listings[0]]) == len(vectorized_data[listings[0]]):
        print("✅ Data lengths match")
    else:
        print("❌ Data lengths differ!")
        print(f"  Iterative: {len(iterative_data[listings[0]])}")
        print(f"  Vectorized: {len(vectorized_data[listings[0]])}")
    
    # Compare sample data
    print("\nSample data comparison (first 5 rows):")
    for listing in listings:
        print(f"\nListing {listing.exchange_id}/{listing.security_id}:")
        print("Iterative bidPrice0:")
        print(iterative_data[listing]['bidPrice0'].head())
        print("Vectorized bidPrice0:")
        print(vectorized_data[listing]['bidPrice0'].head())
    
    # ===================================================================
    # STEP 2: COMPARE BETA VECTORS
    # ===================================================================
    print("\n" + "="*60)
    print("STEP 2: BETA VECTOR COMPARISON")
    print("="*60)
    
    # Get beta histories
    iter_betas = strategy.beta_history
    vec_betas = strategy_vec.beta_history
    
    print(f"Number of beta vectors:")
    print(f"  Iterative: {len(iter_betas)}")
    print(f"  Vectorized: {len(vec_betas)}")
    
    if len(iter_betas) != len(vec_betas):
        print("❌ Different number of beta vectors!")
    else:
        print("✅ Same number of beta vectors")
    
    # Compare beta vectors
    if len(iter_betas) > 0 and len(vec_betas) > 0:
        min_len = min(len(iter_betas), len(vec_betas))
        print(f"\nComparing first {min_len} beta vectors:")
        
        for i in range(min_len):
            iter_beta = np.array(iter_betas[i])
            vec_beta = np.array(vec_betas[i])
            
            # Compare raw beta vectors
            raw_diff = np.abs(iter_beta - vec_beta)
            max_raw_diff = np.max(raw_diff)
            mean_raw_diff = np.mean(raw_diff)
            
            # Compare normalized beta vectors
            iter_norm = iter_beta / np.linalg.norm(iter_beta) if np.linalg.norm(iter_beta) > 1e-10 else iter_beta
            vec_norm = vec_beta / np.linalg.norm(vec_beta) if np.linalg.norm(vec_beta) > 1e-10 else vec_beta
            norm_diff = np.abs(iter_norm - vec_norm)
            max_norm_diff = np.max(norm_diff)
            mean_norm_diff = np.mean(norm_diff)
            
            print(f"Beta {i}:")
            print(f"  Raw - Max diff: {max_raw_diff:.6f}, Mean diff: {mean_raw_diff:.6f}")
            print(f"  Norm - Max diff: {max_norm_diff:.6f}, Mean diff: {mean_norm_diff:.6f}")
            
            if max_raw_diff > 1e-6:
                print(f"  ❌ Raw beta vectors differ significantly")
            else:
                print(f"  ✅ Raw beta vectors match")
    
    # ===================================================================
    # STEP 3: COMPARE SPREAD CALCULATIONS
    # ===================================================================
    print("\n" + "="*60)
    print("STEP 3: SPREAD CALCULATION COMPARISON")
    print("="*60)
    
    # Get precomputed values from vectorized approach
    strategy_hash = str(strategy_vec)
    if strategy_hash in vectorized_backtest.precomputed_values:
        precomputed = vectorized_backtest.precomputed_values[strategy_hash]
        print("✅ Found precomputed values")
        
        # Compare a few specific timestamps
        data_length = len(vectorized_data[listings[0]])
        test_indices = [1000, 2000, 3000]  # Test a few specific points
        
        for test_idx in test_indices:
            if test_idx < data_length:
                print(f"\n--- Testing index {test_idx} ---")
                
                # Vectorized values
                vec_beta = precomputed['beta_vectors'][test_idx]
                vec_zscore = precomputed['z_scores'][test_idx]
                
                print(f"Vectorized beta: {vec_beta}")
                print(f"Vectorized z-score: {vec_zscore}")
                
                # Manual calculation using iterative approach logic
                window_start = max(0, test_idx - strategy.spread_window)
                window_end = test_idx + 1
                
                # Create price matrix for this window
                price_matrix = np.column_stack([
                    np.log(vectorized_data[listing]['bidPrice0'].values[window_start:window_end]) 
                    for listing in listings
                ])
                
                # Calculate spreads using the beta vector
                spreads = price_matrix @ vec_beta.reshape(-1, 1)
                
                if len(spreads) > 1:
                    current_spread = spreads[-1][0]
                    historical_spreads = spreads[:-1].flatten()
                    mean_spread = historical_spreads.mean()
                    std_spread = historical_spreads.std()
                    
                    manual_zscore = (current_spread - mean_spread) / std_spread
                    
                    print(f"Manual calculation:")
                    print(f"  Current spread: {current_spread:.6f}")
                    print(f"  Historical mean: {mean_spread:.6f}")
                    print(f"  Historical std: {std_spread:.6f}")
                    print(f"  Manual z-score: {manual_zscore:.6f}")
                    print(f"  Vectorized z-score: {vec_zscore:.6f}")
                    print(f"  Difference: {abs(manual_zscore - vec_zscore):.6f}")
                    
                    if abs(manual_zscore - vec_zscore) > 1e-6:
                        print(f"  ❌ Z-scores differ!")
                    else:
                        print(f"  ✅ Z-scores match")
    else:
        print("❌ No precomputed values found")
    
    # ===================================================================
    # STEP 4: COMPARE SIGNAL GENERATION
    # ===================================================================
    print("\n" + "="*60)
    print("STEP 4: SIGNAL GENERATION COMPARISON")
    print("="*60)
    
    # Run both backtests to completion
    print("Running iterative research_old to completion...")
    iterative_metrics, iterative_history = iterative_backtest.run()
    
    print("Running vectorized research_old to completion...")
    vectorized_metrics, vectorized_history = vectorized_backtest.run()
    
    # Compare signal histories
    iter_signals = [(s[0], s[1]) for s in iterative_backtest.signal_history]
    vec_signals = [(s[0], s[1]) for s in vectorized_backtest.signal_history]
    
    print(f"Signal counts:")
    print(f"  Iterative: {len(iter_signals)}")
    print(f"  Vectorized: {len(vec_signals)}")
    
    # Compare signal timestamps
    iter_timestamps = set(s[0] for s in iter_signals)
    vec_timestamps = set(s[0] for s in vec_signals)
    
    common_timestamps = iter_timestamps & vec_timestamps
    only_iter = iter_timestamps - vec_timestamps
    only_vec = vec_timestamps - iter_timestamps
    
    print(f"\nTimestamp comparison:")
    print(f"  Common timestamps: {len(common_timestamps)}")
    print(f"  Only in iterative: {len(only_iter)}")
    print(f"  Only in vectorized: {len(only_vec)}")
    
    if len(only_iter) > 0:
        print(f"  Sample timestamps only in iterative: {list(only_iter)[:5]}")
    if len(only_vec) > 0:
        print(f"  Sample timestamps only in vectorized: {list(only_vec)[:5]}")
    
    # Compare signals at common timestamps
    if common_timestamps:
        print(f"\nComparing signals at {len(common_timestamps)} common timestamps:")
        matching_signals = 0
        different_signals = 0
        
        for timestamp in list(common_timestamps)[:10]:  # Check first 10
            iter_signal = next(s[1] for s in iter_signals if s[0] == timestamp)
            vec_signal = next(s[1] for s in vec_signals if s[0] == timestamp)
            
            if (hasattr(iter_signal, 'signal_type') and hasattr(vec_signal, 'signal_type') and
                iter_signal.signal_type == vec_signal.signal_type):
                matching_signals += 1
            else:
                different_signals += 1
                print(f"  Different signal at {timestamp}:")
                print(f"    Iterative: {getattr(iter_signal, 'signal_type', 'N/A')}")
                print(f"    Vectorized: {getattr(vec_signal, 'signal_type', 'N/A')}")
        
        print(f"  Matching signals: {matching_signals}")
        print(f"  Different signals: {different_signals}")
    
    # ===================================================================
    # STEP 5: SUMMARY
    # ===================================================================
    print("\n" + "="*60)
    print("STEP 5: SUMMARY")
    print("="*60)
    
    print("Analysis complete. Check the results above to identify differences.")
    print("Key areas to investigate:")
    print("1. Data alignment and sampling")
    print("2. Beta vector calculation timing")
    print("3. Spread calculation window sizes")
    print("4. Z-score calculation precision")
    print("5. Signal generation logic")

if __name__ == "__main__":
    comprehensive_analysis() 