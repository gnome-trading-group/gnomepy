"""
Example: Hyperparameter Optimization with Optimized Data Loading

This example demonstrates how to use the DataManager and factory function to avoid
repeated data fetching during hyperparameter optimization.
"""

import datetime
from typing import List
import pandas as pd
from gnomepy.data.client import MarketDataClient
from gnomepy.data.types import SchemaType
from gnomepy.backtest.backtest import create_optimized_backtest_factory
from gnomepy.backtest.strategy import CointegrationStrategy
from gnomepy.data.types import Listing


def run_hyperparameter_optimization():
    """Example of hyperparameter optimization with optimized data loading."""
    
    # Initialize your market data client
    client = MarketDataClient()
    
    # Define your time period
    start_datetime = datetime.datetime(2023, 1, 1)
    end_datetime = datetime.datetime(2023, 12, 31)
    
    # Define all listings you might use in your strategies
    all_listings = [
        Listing(exchange_id="BINANCE", security_id="BTCUSDT"),
        Listing(exchange_id="BINANCE", security_id="ETHUSDT"),
        Listing(exchange_id="BINANCE", security_id="ADAUSDT"),
        # Add more listings as needed
    ]
    
    # Create the optimized factory - this preloads ALL data once
    print("Preloading data...")
    data_manager, create_backtest = create_optimized_backtest_factory(
        client=client,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        schema_type=SchemaType.MBP_10,
        all_listings=all_listings
    )
    print("Data preloaded successfully!")
    
    # Define hyperparameter combinations to test
    hyperparameter_combinations = [
        {
            'listings': [all_listings[0], all_listings[1]],  # BTC-ETH pair
            'trade_frequency': 1,
            'beta_refresh_frequency': 1000,
            'spread_window': 100,
            'enter_zscore': 2.0,
            'exit_zscore': 0.3,
            'stop_loss_delta': 0.0,
            'retest_cointegration': False,
            'use_extends': True,
            'use_lob': True,
            'use_dynamic_sizing': True,
            'significance_level': 0.05
        },
        {
            'listings': [all_listings[0], all_listings[2]],  # BTC-ADA pair
            'trade_frequency': 5,
            'beta_refresh_frequency': 500,
            'spread_window': 200,
            'enter_zscore': 1.5,
            'exit_zscore': 0.5,
            'stop_loss_delta': 0.1,
            'retest_cointegration': True,
            'use_extends': True,
            'use_lob': True,
            'use_dynamic_sizing': False,
            'significance_level': 0.01
        },
        # Add more combinations as needed
    ]
    
    results = []
    
    # Run optimization - no data fetching overhead!
    for i, params in enumerate(hyperparameter_combinations):
        print(f"\nTesting combination {i+1}/{len(hyperparameter_combinations)}")
        
        # Create strategy with current parameters
        strategy = CointegrationStrategy(
            listings=params['listings'],
            trade_frequency=params['trade_frequency'],
            beta_refresh_frequency=params['beta_refresh_frequency'],
            spread_window=params['spread_window'],
            enter_zscore=params['enter_zscore'],
            exit_zscore=params['exit_zscore'],
            stop_loss_delta=params['stop_loss_delta'],
            retest_cointegration=params['retest_cointegration'],
            use_extends=params['use_extends'],
            use_lob=params['use_lob'],
            use_dynamic_sizing=params['use_dynamic_sizing'],
            significance_level=params['significance_level']
        )
        
        # Create backtest instance - this is instant since data is preloaded
        backtest = create_backtest(strategies=strategy, use_vectorized=True)
        
        # Run the backtest
        metrics, trade_history = backtest.run()
        
        # Store results
        if not metrics.empty:
            result = {
                'combination': i+1,
                'params': params,
                'total_pnl': metrics.iloc[0]['Total P&L'],
                'win_rate': metrics.iloc[0]['Win Rate'],
                'avg_trade_pnl': metrics.iloc[0]['Avg Trade P&L'],
                'max_drawdown': metrics.iloc[0]['Max Drawdown'],
                'pl_to_dd_ratio': metrics.iloc[0]['Total P&L to Max Drawdown Ratio'],
                'total_trades': metrics.iloc[0]['Total Trades']
            }
            results.append(result)
            
            print(f"  Total P&L: ${result['total_pnl']:.2f}")
            print(f"  Win Rate: {result['win_rate']:.2%}")
            print(f"  Max Drawdown: ${result['max_drawdown']:.2f}")
    
    # Analyze results
    if results:
        results_df = pd.DataFrame(results)
        print("\n" + "="*50)
        print("OPTIMIZATION RESULTS")
        print("="*50)
        
        # Find best performing combination
        best_idx = results_df['total_pnl'].idxmax()
        best_result = results_df.loc[best_idx]
        
        print(f"\nBest combination: {best_result['combination']}")
        print(f"Best Total P&L: ${best_result['total_pnl']:.2f}")
        print(f"Best Win Rate: {best_result['win_rate']:.2%}")
        print(f"Best P&L to Drawdown Ratio: {best_result['pl_to_dd_ratio']:.2f}")
        
        print(f"\nBest parameters:")
        for key, value in best_result['params'].items():
            print(f"  {key}: {value}")
        
        # Show all results sorted by P&L
        print(f"\nAll results (sorted by Total P&L):")
        sorted_results = results_df.sort_values('total_pnl', ascending=False)
        print(sorted_results[['combination', 'total_pnl', 'win_rate', 'max_drawdown', 'pl_to_dd_ratio']].to_string(index=False))
    
    return results


if __name__ == "__main__":
    run_hyperparameter_optimization() 