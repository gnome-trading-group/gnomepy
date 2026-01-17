# Vectorized Backtesting

This document describes the vectorized backtesting functionality that has been added to the gnomepy backtesting system.

## Overview

The vectorized backtest provides a faster alternative to the iterative backtest by precomputing strategy-dependent values (like beta vectors for cointegration strategies) and generating all signals at once, rather than processing each tick individually.

## Key Benefits

1. **Speed**: Significantly faster execution, especially for strategies with complex calculations
2. **Code Reuse**: Reuses existing OMS, signal generation, and portfolio calculation logic
3. **Seamless Toggle**: Easy to switch between iterative and vectorized modes
4. **Same Results**: Produces identical results to the iterative backtest (within numerical precision)

## Architecture

### Base Strategy Class Extensions

The base `Strategy` class has been extended with two new methods:

```python
def precompute_strategy_values(self, listing_data: dict[Listing, pd.DataFrame]) -> dict:
    """Precompute strategy-dependent values for vectorized backtesting."""
    return {}

def generate_vectorized_signals(self, listing_data: dict[Listing, pd.DataFrame], 
                              precomputed_values: dict) -> list[tuple[int, Signal | BasketSignal]]:
    """Generate signals using precomputed values for vectorized backtesting."""
    return []
```

### CointegrationStrategy Implementation

The `CointegrationStrategy` implements these methods to:

1. **Precompute**: Beta vectors, normalized beta vectors, cointegration counts, and z-scores for all timestamps
2. **Generate Signals**: Create all trading signals using the precomputed values

### VectorizedBacktest Class

The `VectorizedBacktest` class inherits from `Backtest` and:

1. Precomputes all strategy values upfront
2. Generates all signals at once
3. Processes signals chronologically through the existing OMS
4. Returns the same metrics as the iterative backtest

## Usage

### Basic Usage

```python
from gnomepy.research_old.backtest import VectorizedBacktest
from gnomepy.research_old.strategy import CointegrationStrategy
from gnomepy.data.types import Listing, SchemaType

# Create strategy
strategy = CointegrationStrategy(
    listings=[Listing(exchange_id=1, security_id=1001), Listing(exchange_id=1, security_id=1002)],
    data_schema_type=SchemaType.MBP_10,
    trade_frequency=10,
    beta_refresh_frequency=1000
)

# Run vectorized research_old
backtest = VectorizedBacktest(
    client=client,
    strategies=[strategy],
    start_datetime=start_datetime,
    end_datetime=end_datetime
)

metrics, history = backtest.run()
```

### Comparison with Iterative Backtest

```python
from gnomepy.research_old.backtest import Backtest, VectorizedBacktest
import time

# Run iterative research_old
start_time = time.time()
iterative_backtest = Backtest(client, strategies, start_datetime, end_datetime)
iterative_metrics, iterative_history = iterative_backtest.run()
iterative_time = time.time() - start_time

# Run vectorized research_old
start_time = time.time()
vectorized_backtest = VectorizedBacktest(client, strategies, start_datetime, end_datetime)
vectorized_metrics, vectorized_history = vectorized_backtest.run()
vectorized_time = time.time() - start_time

print(f"Speedup: {iterative_time / vectorized_time:.2f}x faster")
```

## Performance Characteristics

### When to Use Vectorized Backtest

- **Fast Testing**: When you need to quickly test strategy parameters
- **Large Datasets**: When processing large amounts of historical data
- **Parameter Optimization**: When running many backtests for optimization
- **Strategy Development**: During initial strategy development and testing

### When to Use Iterative Backtest

- **Realistic Simulation**: When you need the most realistic simulation of live trading
- **Latency Testing**: When testing strategy latency characteristics
- **Order Book Simulation**: When detailed order book interaction is important
- **Debugging**: When you need to step through the execution tick by tick

## Implementation Details

### Precomputation Process

1. **Data Preparation**: Load and align all market data
2. **Beta Calculation**: Calculate cointegration betas at specified intervals
3. **Z-Score Calculation**: Compute z-scores for all timestamps with valid betas
4. **Signal Generation**: Generate all trading signals using precomputed values

### Signal Processing

1. **Chronological Sorting**: Sort all generated signals by timestamp
2. **OMS Processing**: Process signals through the existing Order Management System
3. **Portfolio Calculation**: Calculate final portfolio metrics using existing logic

### Memory Usage

The vectorized approach uses more memory upfront to store precomputed values, but processes data much faster. Memory usage scales with:
- Number of timestamps in the dataset
- Number of securities in each strategy
- Complexity of precomputed values

## Extending for New Strategies

To add vectorized support for a new strategy:

1. **Override `precompute_strategy_values`**: Precompute any strategy-dependent values
2. **Override `generate_vectorized_signals`**: Generate signals using precomputed values
3. **Return Timestamps**: Ensure signals include timestamp information for chronological processing

Example:

```python
class MyStrategy(Strategy):
    def precompute_strategy_values(self, listing_data):
        # Precompute your strategy's values
        return {'my_values': computed_values}
    
    def generate_vectorized_signals(self, listing_data, precomputed_values):
        signals = []
        # Generate signals using precomputed values
        for i, timestamp in enumerate(timestamps):
            if should_trade(i):
                signal = create_signal(i)
                signals.append((timestamp, signal))
        return signals
```

## Limitations

1. **Memory Usage**: Higher memory usage due to precomputed values
2. **Real-time Simulation**: Less realistic for real-time trading simulation
3. **Dynamic State**: May not handle complex dynamic state changes as well as iterative approach
4. **Order Book Detail**: May miss some fine-grained order book interactions

## Future Enhancements

Potential improvements to consider:

1. **Hybrid Approach**: Combine vectorized and iterative approaches
2. **Parallel Processing**: Parallelize precomputation across multiple cores
3. **Memory Optimization**: Implement memory-efficient precomputation
4. **Dynamic Precomputation**: Precompute values in chunks to reduce memory usage 