#!/usr/bin/env python3
"""
Simple test script to verify the vectorized research_old implementation.
"""

import unittest
import numpy as np
import pandas as pd
import datetime
from unittest.mock import Mock, MagicMock

from gnomepy.data.types import Listing, SchemaType, Action, SignalType
from gnomepy.research_old.strategy import CointegrationStrategy
from gnomepy.research_old.backtest import VectorizedBacktest
from gnomepy.research_old.trade_signal import TradeSignal, BasketTradeSignal


class TestVectorizedBacktest(unittest.TestCase):
    """Test cases for the vectorized research_old functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.listings = [
            Listing(exchange_id=1, security_id=1001),
            Listing(exchange_id=1, security_id=1002),
        ]
        
        self.strategy = CointegrationStrategy(
            listings=self.listings,
            data_schema_type=SchemaType.MBP_10,
            trade_frequency=10,
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
        
        # Create mock market data
        self.mock_listing_data = {}
        for listing in self.listings:
            # Create 2000 timestamps of mock data
            timestamps = np.arange(1000000000000, 1000000000000 + 2000 * 1000000, 1000000)
            prices = 100 + np.cumsum(np.random.randn(2000) * 0.1)  # Random walk prices
            
            df = pd.DataFrame({
                'timestampEvent': timestamps,
                'bidPrice0': prices,
                'askPrice0': prices + 0.01,
                'bidSize0': np.random.randint(100, 1000, 2000),
                'askSize0': np.random.randint(100, 1000, 2000),
            })
            
            # Add additional order book levels
            for level in range(1, 10):
                df[f'bidPrice{level}'] = prices - level * 0.01
                df[f'askPrice{level}'] = prices + level * 0.01
                df[f'bidSize{level}'] = np.random.randint(50, 500, 2000)
                df[f'askSize{level}'] = np.random.randint(50, 500, 2000)
            
            self.mock_listing_data[listing] = df
    
    def test_precompute_strategy_values(self):
        """Test that strategy values are precomputed correctly."""
        precomputed_values = self.strategy.precompute_strategy_values(self.mock_listing_data)
        
        # Check that all expected keys are present
        expected_keys = ['beta_vectors', 'norm_beta_vectors', 'n_coints_array', 'z_scores', 'timestamps', 'price_matrix']
        for key in expected_keys:
            self.assertIn(key, precomputed_values)
        
        # Check shapes
        data_length = len(self.mock_listing_data[self.listings[0]])
        self.assertEqual(precomputed_values['beta_vectors'].shape, (data_length, len(self.listings)))
        self.assertEqual(precomputed_values['norm_beta_vectors'].shape, (data_length, len(self.listings)))
        self.assertEqual(precomputed_values['n_coints_array'].shape, (data_length,))
        self.assertEqual(precomputed_values['z_scores'].shape, (data_length,))
        self.assertEqual(precomputed_values['timestamps'].shape, (data_length,))
        self.assertEqual(precomputed_values['price_matrix'].shape, (data_length, len(self.listings)))
    
    def test_generate_vectorized_signals(self):
        """Test that vectorized signals are generated correctly."""
        precomputed_values = self.strategy.precompute_strategy_values(self.mock_listing_data)
        signals = self.strategy.generate_vectorized_signals(self.mock_listing_data, precomputed_values)
        
        # Check that signals are returned as (timestamp, signal) tuples
        for signal_tuple in signals:
            self.assertIsInstance(signal_tuple, tuple)
            self.assertEqual(len(signal_tuple), 2)
            timestamp, signal = signal_tuple
            
            # Check timestamp
            self.assertIsInstance(timestamp, (int, np.integer))
            
            # Check signal
            self.assertIsInstance(signal, BasketTradeSignal)
            self.assertEqual(len(signal.signals), len(self.listings))
            self.assertEqual(len(signal.proportions), len(self.listings))
            self.assertEqual(signal.strategy, self.strategy)
            self.assertIn(signal.signal_type, [
                SignalType.ENTER_POSITIVE_MEAN_REVERSION,
                SignalType.ENTER_NEGATIVE_MEAN_REVERSION,
                SignalType.EXIT_POSITIVE_MEAN_REVERSION,
                SignalType.EXIT_NEGATIVE_MEAN_REVERSION
            ])
    
    def test_vectorized_backtest_initialization(self):
        """Test that VectorizedBacktest initializes correctly."""
        # Mock the client
        mock_client = Mock()
        
        start_datetime = datetime.datetime(2023, 1, 1, 9, 30, 0)
        end_datetime = datetime.datetime(2023, 1, 2, 16, 0, 0)
        
        # Mock the _fetch_data method to return our test data
        with unittest.mock.patch.object(VectorizedBacktest, '_fetch_data', return_value=self.mock_listing_data):
            backtest = VectorizedBacktest(
                client=mock_client,
                strategies=[self.strategy],
                start_datetime=start_datetime,
                end_datetime=end_datetime
            )
            
            self.assertEqual(backtest.strategies, [self.strategy])
            self.assertEqual(backtest.start_datetime, start_datetime)
            self.assertEqual(backtest.end_datetime, end_datetime)
            self.assertEqual(backtest.listing_data, self.mock_listing_data)
            self.assertEqual(backtest.precomputed_values, {})
    
    def test_strategy_hash_consistency(self):
        """Test that strategy hashing is consistent."""
        hash1 = hash(self.strategy)
        hash2 = hash(self.strategy)
        self.assertEqual(hash1, hash2)
        
        # Create identical strategy
        strategy2 = CointegrationStrategy(
            listings=self.listings,
            data_schema_type=SchemaType.MBP_10,
            trade_frequency=10,
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
        
        # Should have same hash if equal
        if self.strategy == strategy2:
            self.assertEqual(hash(self.strategy), hash(strategy2))


class TestSignalGeneration(unittest.TestCase):
    """Test cases for signal generation logic."""
    
    def test_signal_creation(self):
        """Test that signals are created with correct attributes."""
        listing = Listing(exchange_id=1, security_id=1001)
        signal = TradeSignal(
            listing=listing,
            action=Action.BUY,
            confidence=1.5
        )
        
        self.assertEqual(signal.listing, listing)
        self.assertEqual(signal.action, Action.BUY)
        self.assertEqual(signal.confidence, 1.5)
    
    def test_basket_signal_creation(self):
        """Test that basket signals are created correctly."""
        listings = [
            Listing(exchange_id=1, security_id=1001),
            Listing(exchange_id=1, security_id=1002),
        ]
        
        strategy = CointegrationStrategy(listings=listings, data_schema_type=SchemaType.MBP_10)
        
        signals = [
            TradeSignal(listing=listings[0], action=Action.BUY, confidence=1.0),
            TradeSignal(listing=listings[1], action=Action.SELL, confidence=1.0),
        ]
        
        proportions = [0.6, -0.4]
        
        basket_signal = BasketTradeSignal(
            signals=signals,
            proportions=proportions,
            strategy=strategy,
            signal_type=SignalType.ENTER_POSITIVE_MEAN_REVERSION
        )
        
        self.assertEqual(len(basket_signal.signals), 2)
        self.assertEqual(basket_signal.proportions, proportions)
        self.assertEqual(basket_signal.strategy, strategy)
        self.assertEqual(basket_signal.signal_type, SignalType.ENTER_POSITIVE_MEAN_REVERSION)


if __name__ == '__main__':
    unittest.main() 