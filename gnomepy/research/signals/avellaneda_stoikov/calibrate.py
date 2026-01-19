#!/usr/bin/env python3
"""
Calibration script for Avellaneda-Stoikov market making signal parameters.

This script loads historical market data and calibrates optimal parameters
for the Avellaneda-Stoikov market making algorithm.

Usage:
    python calibrate_avellaneda_stoikov.py --listing_id 1 --start_date 2025-12-23 --end_date 2025-12-24
"""

import argparse
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize

from gnomepy.data.cached_client import CachedMarketDataClient
from gnomepy.data.types import SchemaType, FIXED_PRICE_SCALE
from gnomepy.registry.api import RegistryClient
from gnomepy.research.signals.avellaneda_stoikov import AvellanedaStoikovSignal


class AvellanedaStoikovCalibrator:
    """Calibrate Avellaneda-Stoikov parameters from historical market data."""
    
    def __init__(
        self,
        listing_id: int,
        start_datetime: datetime.datetime,
        end_datetime: datetime.datetime,
        schema_type: SchemaType = SchemaType.MBP_10,
        market_data_client: Optional[CachedMarketDataClient] = None,
        registry_client: Optional[RegistryClient] = None,
    ):
        """Initialize the calibrator.
        
        Parameters
        ----------
        listing_id : int
            Listing ID to calibrate for
        start_datetime : datetime.datetime
            Start of calibration period
        end_datetime : datetime.datetime
            End of calibration period
        schema_type : SchemaType, default SchemaType.MBP_10
            Market data schema type
        market_data_client : CachedMarketDataClient, optional
            Market data client (will create default if None)
        registry_client : RegistryClient, optional
            Registry client (will create default if None)
        """
        self.listing_id = listing_id
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.schema_type = schema_type
        
        # Initialize clients
        self.market_data_client = market_data_client or CachedMarketDataClient(
            bucket="gnome-market-data-archive-dev",
            aws_profile_name="AWSAdministratorAccess-443370708724"
        )
        self.registry_client = registry_client or RegistryClient(
            api_key="9WPV7CfeqXa578yVYlxdG3kCPFzACr7YaMU0UVma"
        )
        
        # Load listing
        result = self.registry_client.get_listing(listing_id=listing_id)
        if not result:
            raise ValueError(f"Unable to find listing_id: {listing_id}")
        self.listing = result[0]
        
        # Data will be loaded on demand
        self.data_df: Optional[pd.DataFrame] = None
        self.mid_prices: Optional[np.ndarray] = None
        self.bid_prices: Optional[np.ndarray] = None
        self.ask_prices: Optional[np.ndarray] = None
    
    def load_data(self) -> pd.DataFrame:
        """Load and prepare market data.
        
        Returns
        -------
        pd.DataFrame
            Market data with bid/ask prices
        """
        if self.data_df is not None:
            return self.data_df
        
        print(f"Loading market data for listing {self.listing_id}...")
        print(f"  Period: {self.start_datetime} to {self.end_datetime}")
        
        # Load data from market data client
        data_store = self.market_data_client.get_data(
            exchange_id=self.listing.exchange_id,
            security_id=self.listing.security_id,
            start_datetime=self.start_datetime,
            end_datetime=self.end_datetime,
            schema_type=self.schema_type
        )
        
        # Convert to DataFrame
        df = data_store.to_df(price_type="float", size_type="float")
        
        # Extract bid/ask prices
        # For MBP schemas, levels are flattened into bidPrice0-9, askPrice0-9 columns
        if 'bidPrice0' not in df.columns or 'askPrice0' not in df.columns:
            # Try to extract from levels if they exist as objects
            if 'levels' in df.columns:
                levels_list = df['levels'].tolist()
                bid_prices = []
                ask_prices = []
                
                for levels in levels_list:
                    if levels and len(levels) > 0:
                        first_level = levels[0]
                        # Handle both dict and BidAskPair object
                        if isinstance(first_level, dict):
                            bid_px = first_level.get('bid_px', first_level.get('bidPrice0', 0))
                            ask_px = first_level.get('ask_px', first_level.get('askPrice0', 0))
                        else:
                            # BidAskPair object
                            bid_px = first_level.bid_px if hasattr(first_level, 'bid_px') else 0
                            ask_px = first_level.ask_px if hasattr(first_level, 'ask_px') else 0
                        
                        bid_prices.append(bid_px / FIXED_PRICE_SCALE if bid_px > 0 else np.nan)
                        ask_prices.append(ask_px / FIXED_PRICE_SCALE if ask_px > 0 else np.nan)
                    else:
                        bid_prices.append(np.nan)
                        ask_prices.append(np.nan)
                
                df['bidPrice0'] = bid_prices
                df['askPrice0'] = ask_prices
            else:
                raise ValueError("Unable to extract bid/ask prices from data. Expected 'bidPrice0'/'askPrice0' columns or 'levels' column.")
        
        # Calculate mid prices
        df['midPrice'] = (df['bidPrice0'] + df['askPrice0']) / 2.0
        
        # Remove rows with NaN prices
        df = df.dropna(subset=['bidPrice0', 'askPrice0', 'midPrice'])
        
        # Sort by timestamp
        if 'ts_event' in df.columns:
            df = df.sort_values('ts_event')
        elif 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        self.data_df = df
        self.mid_prices = df['midPrice'].values
        self.bid_prices = df['bidPrice0'].values
        self.ask_prices = df['askPrice0'].values
        
        print(f"Loaded {len(df)} market data points")
        print(f"  Price range: ${df['midPrice'].min():.2f} - ${df['midPrice'].max():.2f}")
        
        return df
    
    def calculate_volatility(
        self, 
        window: int = 100, 
        method: str = "log_returns"
    ) -> Tuple[float, np.ndarray]:
        """Calculate volatility per tick from historical prices.
        
        Parameters
        ----------
        window : int, default 100
            Rolling window size
        method : str, default "log_returns"
            Method: "log_returns" or "returns"
        
        Returns
        -------
        float
            Mean volatility per tick
        np.ndarray
            Rolling volatility per tick series
        """
        if self.mid_prices is None:
            self.load_data()
        
        prices = self.mid_prices
        
        if method == "log_returns":
            # Log returns method (standard for volatility)
            log_returns = np.diff(np.log(prices))
            rolling_vol = pd.Series(log_returns).rolling(window=window).std().values
        else:
            # Simple returns method
            returns = np.diff(prices) / prices[:-1]
            rolling_vol = pd.Series(returns).rolling(window=window).std().values
        
        # Mean volatility per tick (excluding NaN values)
        mean_vol = np.nanmean(rolling_vol)
        
        return mean_vol, rolling_vol
    
    def estimate_order_arrival_rate(
        self,
        window: int = 1000,
        method: str = "spread_based"
    ) -> Tuple[float, np.ndarray]:
        """Estimate order arrival rate from market data (in orders per tick).
        
        Parameters
        ----------
        window : int, default 1000
            Window size for rolling estimates
        method : str, default "spread_based"
            Estimation method: "spread_based", "frequency_based", or "combined"
        
        Returns
        -------
        float
            Mean order arrival rate estimate (orders per tick)
        np.ndarray
            Rolling order arrival rate series (orders per tick)
        """
        if self.data_df is None:
            self.load_data()
        
        df = self.data_df
        
        if method == "spread_based":
            # Method 1: Estimate from observed spreads
            # Higher spreads suggest lower arrival rates
            spreads = df['askPrice0'] - df['bidPrice0']
            relative_spreads = spreads / df['midPrice']
            mean_spread = np.nanmean(relative_spreads)
            
            # Estimate k from spread (heuristic: k = 1 / (spread * 10))
            # This gives orders per tick (dimensionless relative to tick frequency)
            k_estimate = 1.0 / (mean_spread * 10.0 + 1e-6)
            rolling_k = 1.0 / (pd.Series(relative_spreads).rolling(window=window).mean().values * 10.0 + 1e-6)
            
        elif method == "frequency_based":
            # Method 2: Estimate from update frequency (orders per tick)
            # For frequency-based, we assume each update is one tick
            # So the arrival rate is simply 1.0 orders per tick
            # (or we can estimate based on actual update frequency if needed)
            k_estimate = 1.0  # Each market update is one tick, so 1 order per tick
            
            # Rolling estimate: count updates per window
            # For simplicity, use 1.0 per tick, but could be refined based on actual frequency
            rolling_k = np.ones(len(df)) * 1.0

        elif method == "combined":
            # Method 3: Combine spread-based and frequency-based estimates
            k_spread, _ = self.estimate_order_arrival_rate(window=window, method="spread_based")
            k_freq, _ = self.estimate_order_arrival_rate(window=window, method="frequency_based")
            
            # Weighted average (can adjust weights)
            k_estimate = 0.6 * k_spread + 0.4 * k_freq
            
            # For rolling, use spread-based (simpler)
            _, rolling_k = self.estimate_order_arrival_rate(window=window, method="spread_based")
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Clamp to reasonable range
        # k_estimate = np.clip(k_estimate, 0.01, 100.0)
        # rolling_k = np.clip(rolling_k, 0.01, 100.0)
        
        return k_estimate, rolling_k
    
    def calculate_optimal_gamma(
        self,
        volatility: float,
        order_arrival_rate: float,
        target_inventory_risk: float = 0.1,
        method: str = "inventory_risk"
    ) -> float:
        """Calculate optimal risk aversion parameter gamma.
        
        Parameters
        ----------
        volatility : float
            Estimated volatility
        order_arrival_rate : float
            Estimated order arrival rate
        target_inventory_risk : float, default 0.1
            Target inventory risk level (for inventory_risk method)
        method : str, default "inventory_risk"
            Calibration method
        
        Returns
        -------
        float
            Optimal gamma value
        """
        if method == "inventory_risk":
            # Gamma based on target inventory risk
            # Higher volatility or lower k → higher gamma (more risk averse)
            # gamma ≈ target_risk / (volatility^2 * k)
            if volatility > 0 and order_arrival_rate > 0:
                gamma = target_inventory_risk / (volatility ** 2 * order_arrival_rate)
                print(f"Gamma: {gamma}")
                print(f"Target inventory risk: {target_inventory_risk}")
                print(f"Volatility: {volatility}")
                print(f"Order arrival rate: {order_arrival_rate}")
            else:
                gamma = 0.1  # Default
        elif method == "spread_based":
            # Gamma based on observed spreads
            # Use relationship from A-S formula
            if self.data_df is None:
                self.load_data()
            
            mean_spread = np.nanmean((self.ask_prices - self.bid_prices) / self.mid_prices)
            # Rough estimate: gamma ≈ spread / (volatility^2)
            print(f"Mean spread: {mean_spread}")
            print(f"Volatility: {volatility}")
            print(f"Order arrival rate: {order_arrival_rate}")
            if volatility > 0:
                gamma = mean_spread / (volatility ** 2 + 1e-6)
                print(f"Gamma: {gamma}")
            else:
                gamma = 0.1
        else:
            gamma = 0.1  # Default
        
        # Clamp to reasonable range
        # gamma = np.clip(gamma, 0.01, 10.0)
        
        return gamma
    
    def calibrate(
        self,
        volatility_window: int = 100,
        optimize_volatility_window: bool = False,
        volatility_window_bounds: Tuple[int, int] = (10, 1000),
        optimize_gamma: bool = True,
        optimize_order_arrival_rate: bool = False,
        gamma_bounds: Tuple[float, float] = (0.01, 10.0),
        order_arrival_rate_bounds: Tuple[float, float] = (0.01, 100.0),
        target_sharpe: float = 1.0,
        target_inventory_risk: float = 0.1,
        order_arrival_rate_method: str = "combined",
    ) -> Dict[str, float]:
        """Calibrate all Avellaneda-Stoikov parameters.
        
        Parameters
        ----------
        volatility_window : int, default 100
            Window size for volatility calculation (used as initial value if optimize_volatility_window=True)
        optimize_volatility_window : bool, default False
            Whether to optimize volatility_window or use provided value
        volatility_window_bounds : Tuple[int, int], default (10, 1000)
            Bounds for volatility_window optimization
        optimize_gamma : bool, default True
            Whether to optimize gamma or use heuristic
        optimize_order_arrival_rate : bool, default False
            Whether to optimize order arrival rate or use estimation
        gamma_bounds : Tuple[float, float], default (0.01, 10.0)
            Bounds for gamma optimization
        order_arrival_rate_bounds : Tuple[float, float], default (0.01, 100.0)
            Bounds for order arrival rate optimization
        target_sharpe : float, default 1.0
            Target Sharpe ratio for optimization
        target_inventory_risk : float, default 0.1
            Target inventory risk level
        order_arrival_rate_method : str, default "combined"
            Method for estimating order arrival rate: "spread_based", "frequency_based", or "combined"
        
        Returns
        -------
        dict
            Dictionary of calibrated parameters
        """
        print("\n=== Calibrating Avellaneda-Stoikov Parameters ===\n")
        
        # Load data
        self.load_data()
        
        # Optimize or use provided volatility window
        if optimize_volatility_window:
            print("1. Optimizing volatility window...")
            optimal_window = self._optimize_volatility_window(
                initial_window=volatility_window,
                bounds=volatility_window_bounds
            )
            volatility_window = optimal_window
            print(f"   Optimal volatility window: {volatility_window}")
        else:
            print(f"1. Using volatility window: {volatility_window}")
        
        # Calculate volatility
        print("\n2. Calculating volatility (per tick)...")
        mean_vol, rolling_vol = self.calculate_volatility(window=volatility_window)
        print(f"   Mean volatility (per tick): {mean_vol:.6f}")
        print(f"   Volatility range (per tick): {np.nanmin(rolling_vol):.6f} - {np.nanmax(rolling_vol):.6f}")
        
        # Estimate or optimize order arrival rate
        print("\n3. Estimating/optimizing order arrival rate (orders per tick)...")
        if optimize_order_arrival_rate:
            print("   Optimizing order arrival rate...")
            # First get initial estimate
            initial_k, rolling_k = self.estimate_order_arrival_rate(method=order_arrival_rate_method)
            print(f"   Initial estimate: {initial_k:.4f} orders/tick")
            
            # Optimize order arrival rate
            optimal_k = self._optimize_order_arrival_rate(
                volatility=mean_vol,
                initial_estimate=initial_k,
                bounds=order_arrival_rate_bounds,
                target_sharpe=target_sharpe
            )
            mean_k = optimal_k
            print(f"   Optimized order arrival rate (k): {mean_k:.4f} orders/tick")
        else:
            mean_k, rolling_k = self.estimate_order_arrival_rate(method=order_arrival_rate_method)
            print(f"   Estimated order arrival rate (k): {mean_k:.4f} orders/tick")
            print(f"   k range: {np.nanmin(rolling_k):.6f} - {np.nanmax(rolling_k):.6f} orders/tick")
        
        # Calculate optimal gamma
        print("\n4. Calculating optimal gamma...")
        if optimize_gamma:
            print("   Optimizing gamma...")
            gamma = self._optimize_gamma(
                volatility=mean_vol,
                order_arrival_rate=mean_k,
                bounds=gamma_bounds,
                target_sharpe=target_sharpe
            )
        else:
            gamma = self.calculate_optimal_gamma(
                volatility=mean_vol,
                order_arrival_rate=mean_k,
                target_inventory_risk=target_inventory_risk,
            )
        print(f"   Optimal gamma: {gamma:.4f}")
        
        # Calculate optimal time horizon
        print("\n5. Calculating optimal time horizon...")
        # Time horizon based on volatility window and data frequency
        time_horizon = self._calculate_time_horizon(volatility_window)
        print(f"   Time horizon: {time_horizon:.2f}")
        
        # Summary
        print("\n=== Calibration Results ===")
        results = {
            'gamma': gamma,
            'volatility_window': volatility_window,
            'time_horizon': time_horizon,
            'order_arrival_rate': mean_k,
            'mean_volatility': mean_vol,
        }
        
        for key, value in results.items():
            print(f"  {key}: {value}")
        
        return results
    
    def _optimize_gamma(
        self,
        volatility: float,
        order_arrival_rate: float,
        bounds: Tuple[float, float] = (0.01, 10.0),
        target_sharpe: float = 1.0
    ) -> float:
        """Optimize gamma using historical data simulation.
        
        Parameters
        ----------
        volatility : float
            Estimated volatility
        order_arrival_rate : float
            Estimated order arrival rate
        bounds : Tuple[float, float]
            Bounds for gamma
        target_sharpe : float
            Target Sharpe ratio
        
        Returns
        -------
        float
            Optimal gamma
        """
        if self.mid_prices is None:
            self.load_data()
        
        # Objective: maximize expected profit while controlling inventory risk
        def objective(gamma_val):
            # Simulate strategy performance with this gamma
            # Use simplified metric: balance between profit and inventory risk
            inventory_risk = gamma_val * (volatility ** 2)
            profit_potential = order_arrival_rate / (gamma_val + 1e-6)
            
            # Combined objective (higher is better)
            # Penalize high inventory risk, reward profit potential
            score = profit_potential - inventory_risk * 10.0
            
            return -score  # Minimize negative score
        
        # Optimize
        result = minimize(
            objective,
            x0=0.1,
            bounds=[bounds],
            method='L-BFGS-B'
        )

        return result.x[0]
    
    def _optimize_volatility_window(
        self,
        initial_window: int = 100,
        bounds: Tuple[int, int] = (10, 1000)
    ) -> int:
        """Optimize volatility window size.
        
        The optimal window balances:
        - Stability: Larger windows give more stable volatility estimates
        - Responsiveness: Smaller windows adapt faster to changing market conditions
        - Statistical significance: Need enough data points for reliable estimates
        
        Parameters
        ----------
        initial_window : int, default 100
            Initial window size to start optimization
        bounds : Tuple[int, int], default (10, 1000)
            Bounds for volatility window optimization
        
        Returns
        -------
        int
            Optimal volatility window size
        """
        if self.mid_prices is None:
            self.load_data()
        
        prices = self.mid_prices
        n_points = len(prices)
        
        # Ensure bounds are reasonable given available data
        min_window = max(bounds[0], 10)  # At least 10 points
        max_window = min(bounds[1], n_points // 2)  # At most half the data
        
        if min_window >= max_window:
            return initial_window
        
        # Test different window sizes
        window_candidates = np.linspace(min_window, max_window, num=20, dtype=int)
        window_candidates = np.unique(window_candidates)  # Remove duplicates
        
        best_window = initial_window
        best_score = -np.inf
        
        scores = []
        
        for window in window_candidates:
            if window >= n_points:
                continue
            
            # Calculate volatility with this window
            _, rolling_vol = self.calculate_volatility(window=int(window))
            
            # Remove NaN values
            valid_vol = rolling_vol[~np.isnan(rolling_vol)]
            
            if len(valid_vol) < 10:
                continue
            
            # Score based on multiple criteria:
            # 1. Stability: Lower volatility of volatility is better
            vol_of_vol = np.std(valid_vol) / (np.mean(valid_vol) + 1e-6)
            
            # 2. Reasonable magnitude: Volatility should be in reasonable range
            mean_vol = np.mean(valid_vol)
            # Penalize if volatility is too small (unrealistic) or too large (unstable)
            if mean_vol < 1e-6:
                magnitude_penalty = 1000.0  # Heavy penalty for unrealistic volatility
            elif mean_vol > 1.0:
                magnitude_penalty = 10.0  # Penalty for very high volatility
            else:
                magnitude_penalty = 0.0
            
            # 3. Coverage: Prefer windows that use more of the data
            coverage = len(valid_vol) / n_points
            
            # Combined score (lower is better, so we negate for maximization)
            # Weight: stability > coverage > magnitude
            score = -vol_of_vol * 10.0 + coverage * 2.0 - magnitude_penalty
            
            scores.append((window, score, vol_of_vol, mean_vol))
            
            if score > best_score:
                best_score = score
                best_window = window
        
        # If no good window found, return initial
        if best_score == -np.inf:
            return initial_window
        
        # Print top candidates for debugging
        if len(scores) > 0:
            scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
            print(f"   Top 3 volatility window candidates:")
            for i, (w, s, vov, mv) in enumerate(scores_sorted[:3]):
                print(f"     Window {w}: score={s:.4f}, vol_of_vol={vov:.6f}, mean_vol={mv:.6f}")
        
        return int(best_window)
    
    def _optimize_order_arrival_rate(
        self,
        volatility: float,
        initial_estimate: float,
        bounds: Tuple[float, float] = (0.01, 100.0),
        target_sharpe: float = 1.0
    ) -> float:
        """Optimize order arrival rate using historical data simulation.
        
        Parameters
        ----------
        volatility : float
            Estimated volatility
        initial_estimate : float
            Initial estimate of order arrival rate
        bounds : Tuple[float, float]
            Bounds for order arrival rate optimization
        target_sharpe : float
            Target Sharpe ratio
        
        Returns
        -------
        float
            Optimal order arrival rate
        """
        if self.mid_prices is None:
            self.load_data()
        
        # Objective: maximize expected profit while maintaining reasonable spreads
        def objective(k_val):
            # For infinite horizon A-S, the spread is: s = gamma * sigma^2 / k + (2/gamma) * ln(1 + gamma/k)
            # We need a reasonable gamma to evaluate this
            # Use a default gamma based on volatility
            default_gamma = 0.1 / (volatility ** 2 + 1e-6)
            default_gamma = np.clip(default_gamma, 0.01, 10.0)
            
            # Calculate expected spread
            spread_component_1 = default_gamma * (volatility ** 2) / (k_val + 1e-6)
            if default_gamma > 0:
                spread_component_2 = (2.0 / default_gamma) * np.log(1.0 + default_gamma / (k_val + 1e-6))
            else:
                spread_component_2 = 0.0
            
            expected_spread = spread_component_1 + spread_component_2
            
            # Calculate profit potential (higher k = more frequent trades = more profit potential)
            profit_potential = k_val
            
            # Calculate spread cost (lower spread = better, but too low = not enough compensation)
            # We want a balance: enough spread to compensate for risk, but not too wide
            if self.data_df is None:
                self.load_data()
            
            # Compare to observed spreads
            observed_spreads = (self.ask_prices - self.bid_prices) / self.mid_prices
            mean_observed_spread = np.nanmean(observed_spreads)
            
            # Penalize if expected spread is too different from observed
            spread_mismatch = abs(expected_spread - mean_observed_spread) / (mean_observed_spread + 1e-6)
            
            # Combined objective (higher is better)
            # Reward profit potential, penalize spread mismatch
            score = profit_potential - spread_mismatch * 10.0
            
            return -score  # Minimize negative score
        
        # Optimize
        result = minimize(
            objective,
            x0=initial_estimate,
            bounds=[bounds],
            method='L-BFGS-B'
        )
        
        optimal_k = np.clip(result.x[0], bounds[0], bounds[1])
        return optimal_k
    
    def _calculate_time_horizon(self, volatility_window: int) -> float:
        """Calculate optimal time horizon.
        
        Parameters
        ----------
        volatility_window : int
            Volatility calculation window
        
        Returns
        -------
        float
            Time horizon in units of volatility_window periods
        """
        # Time horizon should be proportional to volatility window
        # Typical: 1.0 to 5.0 times the window
        # For market making, shorter horizons are better (faster adaptation)
        return 1.0
    
    def get_recommended_parameters(
        self,
        max_inventory: Optional[float] = None
    ) -> Dict[str, float]:
        """Get recommended parameters for AvellanedaStoikovSignal.
        
        Parameters
        ----------
        max_inventory : float, optional
            Maximum inventory size (will be estimated if None)
        
        Returns
        -------
        dict
            Recommended parameters dictionary
        """
        calibrated = self.calibrate()
        
        # Estimate max_inventory if not provided
        if max_inventory is None:
            if self.mid_prices is None:
                self.load_data()
            # Use 2-3x the typical order size as max inventory
            typical_price = np.nanmean(self.mid_prices)
            # Rough estimate: max_inventory = 100 shares or equivalent
            max_inventory = 100.0
        
        recommended = {
            'gamma': calibrated['gamma'],
            'volatility_window': int(calibrated['volatility_window']),
            'time_horizon': calibrated['time_horizon'],
            'order_arrival_rate': calibrated['order_arrival_rate'],
            'max_inventory': max_inventory,
            'liquidation_threshold': 0.8,
            'use_market_orders_for_liquidation': True,
            'trade_frequency': 1,
        }
        
        return recommended


def main():
    """Main entry point for calibration script."""
    parser = argparse.ArgumentParser(
        description="Calibrate Avellaneda-Stoikov market making parameters"
    )
    parser.add_argument(
        '--listing_id',
        type=int,
        required=True,
        help='Listing ID to calibrate for'
    )
    parser.add_argument(
        '--start_date',
        type=str,
        required=True,
        help='Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)'
    )
    parser.add_argument(
        '--end_date',
        type=str,
        required=True,
        help='End date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)'
    )
    parser.add_argument(
        '--schema_type',
        type=str,
        default='mbp-10',
        help='Schema type (default: mbp-10)'
    )
    parser.add_argument(
        '--volatility_window',
        type=int,
        default=100,
        help='Volatility calculation window (default: 100)'
    )
    parser.add_argument(
        '--optimize_gamma',
        action='store_true',
        help='Optimize gamma parameter (default: use heuristic)'
    )
    parser.add_argument(
        '--optimize_volatility_window',
        action='store_true',
        help='Optimize volatility window parameter (default: use provided value)'
    )
    parser.add_argument(
        '--volatility_window_bounds',
        type=int,
        nargs=2,
        default=[10, 1000],
        metavar=('MIN', 'MAX'),
        help='Bounds for volatility window optimization (default: 10 1000)'
    )
    parser.add_argument(
        '--optimize_order_arrival_rate',
        action='store_true',
        help='Optimize order arrival rate parameter (default: use estimation)'
    )
    parser.add_argument(
        '--order_arrival_rate_method',
        type=str,
        default='combined',
        choices=['spread_based', 'frequency_based', 'combined'],
        help='Method for estimating order arrival rate (default: combined)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path for calibrated parameters (JSON)'
    )
    
    args = parser.parse_args()
    
    # Parse dates
    try:
        start_datetime = datetime.datetime.fromisoformat(args.start_date)
    except ValueError:
        start_datetime = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
    
    try:
        end_datetime = datetime.datetime.fromisoformat(args.end_date)
    except ValueError:
        end_datetime = datetime.datetime.strptime(args.end_date, '%Y-%m-%d')
        # If only date provided, set to end of day
        end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
    
    # Parse schema type
    schema_type = SchemaType(args.schema_type)
    
    # Create calibrator
    calibrator = AvellanedaStoikovCalibrator(
        listing_id=args.listing_id,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        schema_type=schema_type
    )
    
    # Run calibration
    results = calibrator.calibrate(
        volatility_window=args.volatility_window,
        optimize_volatility_window=args.optimize_volatility_window,
        volatility_window_bounds=tuple(args.volatility_window_bounds),
        optimize_gamma=args.optimize_gamma,
        optimize_order_arrival_rate=args.optimize_order_arrival_rate,
        order_arrival_rate_method=args.order_arrival_rate_method
    )
    
    # Get recommended parameters
    recommended = calibrator.get_recommended_parameters()
    
    print("\n=== Recommended Parameters for AvellanedaStoikovSignal ===")
    print("\nUsage example:")
    print(f"signal = AvellanedaStoikovSignal(")
    print(f"    listing=listing,")
    print(f"    data_schema_type=SchemaType.{schema_type.name},")
    for key, value in recommended.items():
        if isinstance(value, bool):
            print(f"    {key}={value},")
        elif isinstance(value, int):
            print(f"    {key}={value},")
        elif isinstance(value, float):
            print(f"    {key}={value:.4f},")
        else:
            print(f"    {key}={value},")
    print(")")
    
    # Save to file if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(recommended, f, indent=2)
        print(f"\nParameters saved to {args.output}")


if __name__ == "__main__":
    main()

