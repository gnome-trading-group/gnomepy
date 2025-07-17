#!/usr/bin/env python3
"""
Example script demonstrating the use of VectorizedBacktest vs regular Backtest.

This script shows how to run the same strategy using both the iterative and vectorized
approaches, allowing for easy comparison of performance and results.
"""

import datetime
from gnomepy.data.client import MarketDataClient
from gnomepy.data.types import Listing, SchemaType
from gnomepy.research_old.strategy import CointegrationStrategy
from gnomepy.research_old.backtest import Backtest, VectorizedBacktest
from gnomepy.research_old.trade_signal import TradeSignal, BasketTradeSignal
import time
import numpy as np
import pandas as pd

def run_comparison_example():
    """Run a comparison between iterative and vectorized backtesting."""
    
    # Initialize market data client
    client = MarketDataClient(bucket="gnome-market-data-prod", aws_profile_name="AWSAdministratorAccess-241533121172")
    
    # Define test parameters
    start_datetime = datetime.datetime(2025, 6, 18)
    end_datetime = datetime.datetime(2025, 6, 21)
    
    # Create test listings (you'll need to replace with actual exchange/security IDs)
    listings = [
        Listing(exchange_id=4, security_id=1),
        Listing(exchange_id=1, security_id=1),
    ]
    
    # Create strategy
    strategy = CointegrationStrategy(
        listings=listings,
        data_schema_type=SchemaType.MBP_10,
        trade_frequency=50,  # Trade every 10 ticks
        beta_refresh_frequency=5000,  # Refresh betas every 1000 ticks
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
    start_time = time.time()
    
    # Run iterative research_old
    iterative_backtest = Backtest(
        client=client,
        strategies=[strategy],
        start_datetime=start_datetime,
        end_datetime=end_datetime
    )
    
    iterative_metrics, iterative_history = iterative_backtest.run()
    iterative_time = time.time() - start_time
    
    print(f"Iterative research_old completed in {iterative_time:.2f} seconds")
    print("Iterative research_old metrics:")
    print(iterative_metrics)
    print()
    
    print("=== Running Vectorized Backtest ===")
    start_time = time.time()
    
    # Run vectorized research_old
    vectorized_backtest = VectorizedBacktest(
        client=client,
        strategies=[strategy],
        start_datetime=start_datetime,
        end_datetime=end_datetime
    )
    
    vectorized_metrics, vectorized_history = vectorized_backtest.run()
    vectorized_time = time.time() - start_time
    
    print(f"Vectorized research_old completed in {vectorized_time:.2f} seconds")
    print("Vectorized research_old metrics:")
    print(vectorized_metrics)
    print()
    
    # Compare results
    print("=== Performance Comparison ===")
    speedup = iterative_time / vectorized_time if vectorized_time > 0 else float('inf')
    print(f"Speedup: {speedup:.2f}x faster")
    print()
    
    print("=== Results Comparison ===")
    if not iterative_metrics.empty and not vectorized_metrics.empty:
        print("Total P&L comparison:")
        print(f"  Iterative:  {iterative_metrics['total_pl'].iloc[0]:.2f}")
        print(f"  Vectorized: {vectorized_metrics['total_pl'].iloc[0]:.2f}")
        print(f"  Difference: {abs(iterative_metrics['total_pl'].iloc[0] - vectorized_metrics['total_pl'].iloc[0]):.2f}")
        
        print("\nTrade count comparison:")
        print(f"  Iterative:  {iterative_metrics['total_trades'].iloc[0]}")
        print(f"  Vectorized: {vectorized_metrics['total_trades'].iloc[0]}")
    else:
        print("No trades generated in one or both backtests")

    # Compare signal histories
    print("\n=== Signal History Comparison ===")
    
    # Convert signal histories to DataFrames for easier comparison
    iter_signals_df = pd.DataFrame(iterative_backtest.signal_history, 
                                 columns=['timestamp', 'signal'])
    vec_signals_df = pd.DataFrame(vectorized_backtest.signal_history,
                                columns=['timestamp', 'signal'])
    
    # Sort by timestamp
    iter_signals_df = iter_signals_df.sort_values('timestamp')
    vec_signals_df = vec_signals_df.sort_values('timestamp')
    
    print(f"\nTotal signals generated:")
    print(f"  Iterative:  {len(iter_signals_df)}")
    print(f"  Vectorized: {len(vec_signals_df)}")
    
    # Compare signal timestamps
    common_timestamps = set(iter_signals_df['timestamp']).intersection(set(vec_signals_df['timestamp']))
    print(f"\nSignals at matching timestamps: {len(common_timestamps)}")
    
    # Compare signal details at matching timestamps
    if common_timestamps:
        print("\nAnalyzing signals at matching timestamps...")
        matched_signals = pd.merge(iter_signals_df, vec_signals_df, 
                                 on='timestamp', suffixes=('_iter', '_vec'))
        
        # Count signals with matching properties
        matching_signals = 0
        for _, row in matched_signals.iterrows():
            iter_signal = row['signal_iter']
            vec_signal = row['signal_vec']
            if (isinstance(iter_signal, BasketTradeSignal) and 
                isinstance(vec_signal, BasketTradeSignal)):
                if (iter_signal.listings == vec_signal.listings and 
                    iter_signal.quantities == vec_signal.quantities and
                    iter_signal.side == vec_signal.side):
                    matching_signals += 1
                    
        print(f"Signals with matching properties: {matching_signals}")
        print(f"Signals with different properties: {len(matched_signals) - matching_signals}")

    # Get beta vectors from each research_old
    # Get beta history from each strategy
    iterative_betas = []
    for strategy in iterative_backtest.strategies:
        if isinstance(strategy, CointegrationStrategy):
            iterative_betas.append(strategy.beta_history)
            
    vectorized_betas = []
    for strategy in vectorized_backtest.strategies:
        if isinstance(strategy, CointegrationStrategy):
            strategy_hash = str(strategy)
            if strategy_hash in vectorized_backtest.precomputed_values:
                vectorized_betas.append(strategy.beta_history)
    # Compare beta vectors if both have data
    if iterative_betas and vectorized_betas:
        print("\n=== Beta Vector Comparison ===")
        for i, (iter_beta, vec_beta) in enumerate(zip(iterative_betas, vectorized_betas)):
            if iter_beta and vec_beta:  # Check if beta histories exist
                # Create DataFrames with timestamps and betas
                # Print any beta vectors that result in singular matrices
                for b in iter_beta:
                    if np.linalg.norm(b) <= 1e-10:
                        print(f"Singular matrix detected in iterative beta: {b}")
                for b in vec_beta:
                    if np.linalg.norm(b) <= 1e-10:
                        print(f"Singular matrix detected in vectorized beta: {b}")

                iter_df = pd.DataFrame({
                    'timestamp': strategy.beta_timestamps,
                    'beta': iter_beta,
                    'beta_normalized': [b/np.linalg.norm(b) if np.linalg.norm(b) > 1e-10 else b for b in iter_beta]
                })
                vec_df = pd.DataFrame({
                    'timestamp': strategy.beta_timestamps,
                    'beta': vec_beta,
                    'beta_normalized': [b/np.linalg.norm(b) if np.linalg.norm(b) > 1e-10 else b for b in vec_beta]
                })
                # Join on timestamp
                comparison_df = pd.merge(
                    iter_df, 
                    vec_df,
                    on='timestamp',
                    suffixes=('_iter', '_vec')
                )
                pd.set_option('display.max_rows', None)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                print(iter_df)
                print(vec_df)
                print(comparison_df)
                pd.reset_option('display.max_rows')
                pd.reset_option('display.max_columns') 
                pd.reset_option('display.width')
                
                # Calculate differences
                # diff = np.abs(comparison_df['beta_iter'] - comparison_df['beta_vec'])
                # print(f"\nStrategy {i+1}:")
                # print(f"Max difference: {np.max(diff):.6f}")
                # print(f"Mean difference: {np.mean(diff):.6f}")
                # print(f"Std difference: {np.std(diff):.6f}")
                
                # Compare number of matched timestamps
                print(f"\nMatched timestamps: {len(comparison_df)} out of {len(iter_df)} iterative and {len(vec_df)} vectorized")

def run_vectorized_only_example():
    """Run only the vectorized research_old for quick testing."""
    
    # Initialize market data client
    client = MarketDataClient(bucket="gnome-market-data-prod", aws_profile_name="AWSAdministratorAccess-241533121172")
    
    # Define test parameters
    start_datetime = datetime.datetime(2025, 6, 18)
    end_datetime = datetime.datetime(2025, 6, 21)
    
    # Create test listings (you'll need to replace with actual exchange/security IDs)
    listings = [
        Listing(exchange_id=4, security_id=1),
        Listing(exchange_id=7, security_id=1),
    ]
    
    # Create strategy
    strategy = CointegrationStrategy(
        listings=listings,
        data_schema_type=SchemaType.MBP_10,
        trade_frequency=1000,  # Trade every 10 ticks
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
    
    print("=== Running Vectorized Backtest ===")
    start_time = time.time()
    
    # Run vectorized research_old
    vectorized_backtest = VectorizedBacktest(
        client=client,
        strategies=[strategy],
        start_datetime=start_datetime,
        end_datetime=end_datetime
    )
    
    metrics, history = vectorized_backtest.run()
    total_time = time.time() - start_time
    
    print(f"Vectorized research_old completed in {total_time:.2f} seconds")
    print("Metrics:")
    print(metrics)
    
    if not history.empty:
        print(f"\nGenerated {len(history)} trades")
        print("Sample trades:")
        print(history.head())

if __name__ == "__main__":
    # Uncomment the function you want to run
    # run_comparison_example()  # Run both iterative and vectorized for comparison
    run_vectorized_only_example()  # Run only vectorized for quick testing 