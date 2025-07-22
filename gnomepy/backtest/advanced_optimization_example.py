"""
Advanced Example: Hyperparameter Optimization with Optuna and Optimized Data Loading

This example demonstrates how to integrate the optimized backtest factory with
popular optimization libraries like Optuna for more sophisticated hyperparameter tuning.
"""

import datetime
import optuna
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from gnomepy.data.client import MarketDataClient
from gnomepy.data.types import SchemaType
from gnomepy.backtest.backtest import create_optimized_backtest_factory
from gnomepy.backtest.strategy import CointegrationStrategy
from gnomepy.data.types import Listing


class OptimizedBacktestObjective:
    """Objective function for Optuna optimization with preloaded data."""
    
    def __init__(self, create_backtest_func, all_listings: List[Listing], 
                 optimization_metric: str = 'total_pnl'):
        """Initialize the objective function.
        
        Args:
            create_backtest_func: Factory function to create backtest instances
            all_listings: List of all available listings
            optimization_metric: Metric to optimize ('total_pnl', 'sharpe_ratio', 'calmar_ratio')
        """
        self.create_backtest = create_backtest_func
        self.all_listings = all_listings
        self.optimization_metric = optimization_metric
        
    def __call__(self, trial: optuna.Trial) -> float:
        """Objective function called by Optuna.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            float: Objective value to minimize (negative for maximization)
        """
        # Define hyperparameter search space
        params = self._suggest_hyperparameters(trial)
        
        try:
            # Create strategy with suggested parameters
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
            
            # Create and run backtest
            backtest = self.create_backtest(strategies=strategy, use_vectorized=True)
            metrics, trade_history = backtest.run()
            
            if metrics.empty:
                return float('-inf')  # Penalize failed runs
            
            # Calculate objective value based on optimization metric
            objective_value = self._calculate_objective(metrics.iloc[0])
            
            # Report intermediate values for monitoring
            trial.report(objective_value, step=0)
            
            return objective_value
            
        except Exception as e:
            print(f"Trial failed with error: {e}")
            return float('-inf')  # Penalize failed runs
    
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for the current trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dict: Hyperparameters for the trial
        """
        # Choose listing pair
        pair_idx = trial.suggest_categorical('pair_idx', [0, 1, 2])  # Assuming 3 pairs
        if pair_idx == 0:
            listings = [self.all_listings[0], self.all_listings[1]]  # BTC-ETH
        elif pair_idx == 1:
            listings = [self.all_listings[0], self.all_listings[2]]  # BTC-ADA
        else:
            listings = [self.all_listings[1], self.all_listings[2]]  # ETH-ADA
        
        return {
            'listings': listings,
            'trade_frequency': trial.suggest_int('trade_frequency', 1, 10),
            'beta_refresh_frequency': trial.suggest_int('beta_refresh_frequency', 100, 2000),
            'spread_window': trial.suggest_int('spread_window', 50, 500),
            'enter_zscore': trial.suggest_float('enter_zscore', 1.0, 3.0),
            'exit_zscore': trial.suggest_float('exit_zscore', 0.1, 1.0),
            'stop_loss_delta': trial.suggest_float('stop_loss_delta', 0.0, 0.5),
            'retest_cointegration': trial.suggest_categorical('retest_cointegration', [True, False]),
            'use_extends': trial.suggest_categorical('use_extends', [True, False]),
            'use_lob': trial.suggest_categorical('use_lob', [True, False]),
            'use_dynamic_sizing': trial.suggest_categorical('use_dynamic_sizing', [True, False]),
            'significance_level': trial.suggest_categorical('significance_level', [0.01, 0.05, 0.1])
        }
    
    def _calculate_objective(self, metrics: pd.Series) -> float:
        """Calculate objective value based on optimization metric.
        
        Args:
            metrics: Strategy performance metrics
            
        Returns:
            float: Objective value (negative for maximization)
        """
        if self.optimization_metric == 'total_pnl':
            return -metrics['Total P&L']  # Negative for maximization
        elif self.optimization_metric == 'sharpe_ratio':
            # Approximate Sharpe ratio using P&L ratio
            return -metrics['P&L Ratio']
        elif self.optimization_metric == 'calmar_ratio':
            # Calmar ratio = Total P&L / Max Drawdown
            return -metrics['Total P&L to Max Drawdown Ratio']
        else:
            return -metrics['Total P&L']


def run_optuna_optimization(n_trials: int = 100, optimization_metric: str = 'total_pnl'):
    """Run hyperparameter optimization using Optuna.
    
    Args:
        n_trials: Number of optimization trials
        optimization_metric: Metric to optimize
        
    Returns:
        optuna.Study: Completed optimization study
    """
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
    
    # Create objective function
    objective = OptimizedBacktestObjective(
        create_backtest_func=create_backtest,
        all_listings=all_listings,
        optimization_metric=optimization_metric
    )
    
    # Create and run optimization study
    study = optuna.create_study(
        direction='minimize',  # We minimize the negative objective
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )
    
    print(f"Starting optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study


def analyze_optimization_results(study: optuna.Study):
    """Analyze and display optimization results.
    
    Args:
        study: Completed Optuna study
    """
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    
    # Best trial
    best_trial = study.best_trial
    print(f"\nBest trial: {best_trial.number}")
    print(f"Best objective value: {-best_trial.value:.2f}")  # Convert back to positive
    
    print(f"\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Optimization history
    print(f"\nOptimization history:")
    print(f"  Number of trials: {len(study.trials)}")
    print(f"  Number of completed trials: {len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))}")
    print(f"  Number of pruned trials: {len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))}")
    print(f"  Number of failed trials: {len(study.get_trials(states=[optuna.trial.TrialState.FAIL]))}")
    
    # Parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        print(f"\nParameter importance:")
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {param}: {imp:.3f}")
    except Exception as e:
        print(f"Could not calculate parameter importance: {e}")
    
    # Optimization plot (if matplotlib is available)
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Optimization history
        optuna.visualization.matplotlib.plot_optimization_history(study, ax=ax1)
        ax1.set_title('Optimization History')
        
        # Parameter importance
        optuna.visualization.matplotlib.plot_param_importances(study, ax=ax2)
        ax2.set_title('Parameter Importance')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")


if __name__ == "__main__":
    # Run optimization
    study = run_optuna_optimization(n_trials=50, optimization_metric='total_pnl')
    
    # Analyze results
    analyze_optimization_results(study) 