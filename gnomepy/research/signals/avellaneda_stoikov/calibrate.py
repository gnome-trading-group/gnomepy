#!/usr/bin/env python3
"""
Calibration script for Avellaneda-Stoikov market making signal parameters.

Usage:
    python calibrate.py --listing_id 1 --start_date 2025-12-23 --end_date 2025-12-24
"""

import argparse
import datetime
import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

from gnomepy.data.cached_client import CachedMarketDataClient
from gnomepy.data.types import SchemaType, FIXED_PRICE_SCALE
from gnomepy.registry.api import RegistryClient
from gnomepy.research.signals.avellaneda_stoikov import AvellanedaStoikovSignal

logger = logging.getLogger(__name__)


class AvellanedaStoikovCalibrator:
    """Calibrate Avellaneda-Stoikov parameters from historical market data."""

    def __init__(
        self,
        listing_id: int,
        start_datetime: datetime.datetime,
        end_datetime: datetime.datetime,
        schema_type: SchemaType = SchemaType.MBP_10,
        market_data_client: CachedMarketDataClient = None,
        registry_client: RegistryClient = None,
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
        market_data_client : CachedMarketDataClient
            Market data client
        registry_client : RegistryClient
            Registry client
        """
        self.listing_id = listing_id
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.schema_type = schema_type

        if market_data_client is None:
            raise ValueError("market_data_client is required")
        if registry_client is None:
            raise ValueError("registry_client is required")

        self.market_data_client = market_data_client
        self.registry_client = registry_client

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
        """Load and prepare market data."""
        if self.data_df is not None:
            return self.data_df

        logger.info("Loading market data for listing %d...", self.listing_id)
        logger.info("  Period: %s to %s", self.start_datetime, self.end_datetime)

        data_store = self.market_data_client.get_data(
            exchange_id=self.listing.exchange_id,
            security_id=self.listing.security_id,
            start_datetime=self.start_datetime,
            end_datetime=self.end_datetime,
            schema_type=self.schema_type
        )

        df = data_store.to_df(price_type="float", size_type="float")

        if 'bidPrice0' not in df.columns or 'askPrice0' not in df.columns:
            if 'levels' in df.columns:
                levels_list = df['levels'].tolist()
                bid_prices = []
                ask_prices = []

                for levels in levels_list:
                    if levels and len(levels) > 0:
                        first_level = levels[0]
                        if isinstance(first_level, dict):
                            bid_px = first_level.get('bid_px', first_level.get('bidPrice0', 0))
                            ask_px = first_level.get('ask_px', first_level.get('askPrice0', 0))
                        else:
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

        df['midPrice'] = (df['bidPrice0'] + df['askPrice0']) / 2.0
        df = df.dropna(subset=['bidPrice0', 'askPrice0', 'midPrice'])

        if 'ts_event' in df.columns:
            df = df.sort_values('ts_event')
        elif 'timestamp' in df.columns:
            df = df.sort_values('timestamp')

        self.data_df = df
        self.mid_prices = df['midPrice'].values
        self.bid_prices = df['bidPrice0'].values
        self.ask_prices = df['askPrice0'].values

        logger.info("Loaded %d market data points", len(df))
        logger.info("  Price range: $%.2f - $%.2f", df['midPrice'].min(), df['midPrice'].max())

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
            log_returns = np.diff(np.log(prices))
            rolling_vol = pd.Series(log_returns).rolling(window=window).std().values
        else:
            returns = np.diff(prices) / prices[:-1]
            rolling_vol = pd.Series(returns).rolling(window=window).std().values

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
            spreads = df['askPrice0'] - df['bidPrice0']
            relative_spreads = spreads / df['midPrice']
            mean_spread = np.nanmean(relative_spreads)

            k_estimate = 1.0 / (mean_spread * 10.0 + 1e-6)
            rolling_k = 1.0 / (pd.Series(relative_spreads).rolling(window=window).mean().values * 10.0 + 1e-6)

        elif method == "frequency_based":
            k_estimate = 1.0
            rolling_k = np.ones(len(df)) * 1.0

        elif method == "combined":
            k_spread, _ = self.estimate_order_arrival_rate(window=window, method="spread_based")
            k_freq, _ = self.estimate_order_arrival_rate(window=window, method="frequency_based")

            k_estimate = 0.6 * k_spread + 0.4 * k_freq
            _, rolling_k = self.estimate_order_arrival_rate(window=window, method="spread_based")

        else:
            raise ValueError(f"Unknown method: {method}")

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
            Calibration method: "inventory_risk" or "spread_based"

        Returns
        -------
        float
            Optimal gamma value
        """
        if method == "inventory_risk":
            if volatility > 0 and order_arrival_rate > 0:
                gamma = target_inventory_risk / (volatility ** 2 * order_arrival_rate)
                logger.debug("Gamma: %.4f (target_risk=%.4f, vol=%.6f, k=%.4f)",
                             gamma, target_inventory_risk, volatility, order_arrival_rate)
            else:
                gamma = 0.1
        elif method == "spread_based":
            if self.data_df is None:
                self.load_data()

            relative_spread = np.nanmean((self.ask_prices - self.bid_prices) / self.mid_prices)
            avg_mid_price = np.nanmean(self.mid_prices)

            logger.debug("Mean relative spread: %.6f, avg mid: $%.2f", relative_spread, avg_mid_price)

            if volatility > 0 and order_arrival_rate > 0:
                gamma = (relative_spread * avg_mid_price * order_arrival_rate) / (volatility ** 2 + 1e-6)
                logger.debug("Gamma: %.4f", gamma)
            else:
                gamma = 0.1
        else:
            gamma = 0.1

        return gamma

    def _optimize_volatility_window(
        self,
        initial_window: int = 100,
        bounds: Tuple[int, int] = (10, 1000)
    ) -> int:
        """Optimize volatility window size.

        Balances stability (larger windows) vs responsiveness (smaller windows).
        """
        if self.mid_prices is None:
            self.load_data()

        prices = self.mid_prices
        n_points = len(prices)

        min_window = max(bounds[0], 10)
        max_window = min(bounds[1], n_points // 2)

        if min_window >= max_window:
            return initial_window

        window_candidates = np.linspace(min_window, max_window, num=20, dtype=int)
        window_candidates = np.unique(window_candidates)

        best_window = initial_window
        best_score = -np.inf

        scores = []

        for window in window_candidates:
            if window >= n_points:
                continue

            _, rolling_vol = self.calculate_volatility(window=int(window))
            valid_vol = rolling_vol[~np.isnan(rolling_vol)]

            if len(valid_vol) < 10:
                continue

            vol_of_vol = np.std(valid_vol) / (np.mean(valid_vol) + 1e-6)
            mean_vol = np.mean(valid_vol)

            if mean_vol < 1e-6:
                magnitude_penalty = 1000.0
            elif mean_vol > 1.0:
                magnitude_penalty = 10.0
            else:
                magnitude_penalty = 0.0

            coverage = len(valid_vol) / n_points
            score = -vol_of_vol * 10.0 + coverage * 2.0 - magnitude_penalty

            scores.append((window, score, vol_of_vol, mean_vol))

            if score > best_score:
                best_score = score
                best_window = window

        if best_score == -np.inf:
            return initial_window

        if len(scores) > 0:
            scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
            logger.debug("Top 3 volatility window candidates:")
            for w, s, vov, mv in scores_sorted[:3]:
                logger.debug("  Window %d: score=%.4f, vol_of_vol=%.6f, mean_vol=%.6f", w, s, vov, mv)

        return int(best_window)

    def calibrate(
        self,
        volatility_window: int = 100,
        optimize_volatility_window: bool = False,
        volatility_window_bounds: Tuple[int, int] = (10, 1000),
        target_inventory_risk: float = 0.1,
        order_arrival_rate_method: str = "combined",
    ) -> Dict[str, float]:
        """Calibrate all Avellaneda-Stoikov parameters.

        Parameters
        ----------
        volatility_window : int, default 100
            Window size for volatility calculation
        optimize_volatility_window : bool, default False
            Whether to optimize volatility_window or use provided value
        volatility_window_bounds : Tuple[int, int], default (10, 1000)
            Bounds for volatility_window optimization
        target_inventory_risk : float, default 0.1
            Target inventory risk level
        order_arrival_rate_method : str, default "combined"
            Method for estimating order arrival rate

        Returns
        -------
        dict
            Dictionary of calibrated parameters
        """
        logger.info("Calibrating Avellaneda-Stoikov parameters")

        self.load_data()

        if optimize_volatility_window:
            logger.info("Optimizing volatility window...")
            volatility_window = self._optimize_volatility_window(
                initial_window=volatility_window,
                bounds=volatility_window_bounds
            )
            logger.info("Optimal volatility window: %d", volatility_window)
        else:
            logger.info("Using volatility window: %d", volatility_window)

        logger.info("Calculating volatility (per tick)...")
        mean_vol, rolling_vol = self.calculate_volatility(window=volatility_window)
        logger.info("Mean volatility: %.6f, range: %.6f - %.6f",
                     mean_vol, np.nanmin(rolling_vol), np.nanmax(rolling_vol))

        logger.info("Estimating order arrival rate...")
        mean_k, rolling_k = self.estimate_order_arrival_rate(method=order_arrival_rate_method)
        logger.info("Order arrival rate (k): %.4f, range: %.6f - %.6f",
                     mean_k, np.nanmin(rolling_k), np.nanmax(rolling_k))

        logger.info("Calculating optimal gamma...")
        gamma = self.calculate_optimal_gamma(
            volatility=mean_vol,
            order_arrival_rate=mean_k,
            target_inventory_risk=target_inventory_risk,
        )
        logger.info("Optimal gamma: %.4f", gamma)

        results = {
            'gamma': gamma,
            'volatility_window': volatility_window,
            'order_arrival_rate': mean_k,
            'mean_volatility': mean_vol,
        }

        logger.info("Calibration results: %s", results)

        return results

    def get_recommended_parameters(
        self,
        max_inventory: Optional[float] = None
    ) -> Dict[str, float]:
        """Get recommended parameters for AvellanedaStoikovSignal."""
        calibrated = self.calibrate()

        if max_inventory is None:
            max_inventory = 100.0

        recommended = {
            'gamma': calibrated['gamma'],
            'volatility_window': int(calibrated['volatility_window']),
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
    parser.add_argument('--listing_id', type=int, required=True, help='Listing ID to calibrate for')
    parser.add_argument('--start_date', type=str, required=True, help='Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end_date', type=str, required=True, help='End date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--schema_type', type=str, default='mbp-10', help='Schema type (default: mbp-10)')
    parser.add_argument('--volatility_window', type=int, default=100, help='Volatility calculation window (default: 100)')
    parser.add_argument('--optimize_volatility_window', action='store_true', help='Optimize volatility window')
    parser.add_argument('--volatility_window_bounds', type=int, nargs=2, default=[10, 1000], metavar=('MIN', 'MAX'))
    parser.add_argument('--order_arrival_rate_method', type=str, default='combined',
                        choices=['spread_based', 'frequency_based', 'combined'])
    parser.add_argument('--output', type=str, help='Output file path for calibrated parameters (JSON)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        start_datetime = datetime.datetime.fromisoformat(args.start_date)
    except ValueError:
        start_datetime = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')

    try:
        end_datetime = datetime.datetime.fromisoformat(args.end_date)
    except ValueError:
        end_datetime = datetime.datetime.strptime(args.end_date, '%Y-%m-%d')
        end_datetime = end_datetime.replace(hour=23, minute=59, second=59)

    schema_type = SchemaType(args.schema_type)

    market_data_client = CachedMarketDataClient(
        bucket=os.environ["GNOME_MARKET_DATA_BUCKET"],
        aws_profile_name=os.environ.get("AWS_PROFILE"),
    )
    registry_client = RegistryClient(
        api_key=os.environ["GNOME_REGISTRY_API_KEY"],
    )

    calibrator = AvellanedaStoikovCalibrator(
        listing_id=args.listing_id,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        schema_type=schema_type,
        market_data_client=market_data_client,
        registry_client=registry_client,
    )

    results = calibrator.calibrate(
        volatility_window=args.volatility_window,
        optimize_volatility_window=args.optimize_volatility_window,
        volatility_window_bounds=tuple(args.volatility_window_bounds),
        order_arrival_rate_method=args.order_arrival_rate_method
    )

    recommended = calibrator.get_recommended_parameters()

    logger.info("Recommended parameters for AvellanedaStoikovSignal:")
    logger.info("signal = AvellanedaStoikovSignal(")
    logger.info("    listing=listing,")
    logger.info("    data_schema_type=SchemaType.%s,", schema_type.name)
    for key, value in recommended.items():
        if isinstance(value, float):
            logger.info("    %s=%.4f,", key, value)
        else:
            logger.info("    %s=%s,", key, value)
    logger.info(")")

    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(recommended, f, indent=2)
        logger.info("Parameters saved to %s", args.output)


if __name__ == "__main__":
    main()
