# MBPSimulatedExchange

A high-frequency trading (HFT) simulated exchange implementation that supports Market By Price (MBP) data and realistic order matching with queue position simulation.

## Features

### Order Types
- **LIMIT Orders**: Traditional limit orders with price and size
- **MARKET Orders**: Market orders that execute immediately at best available prices

### Time In Force (TIF) Support
- **GTC (Good Till Cancelled)**: Orders remain active until cancelled
- **GTX (Good Till Crossing)**: Orders remain active until market crossing
- **IOC (Immediate or Cancel)**: Orders must execute immediately or be cancelled
- **FOK (Fill or Kill)**: Orders must execute completely or be cancelled

### Queue Position Simulation
The exchange supports realistic queue position modeling through the `QueueModel` interface:

- **SimpleQueueModel**: Basic FIFO behavior with position-based execution probability
- **RealisticQueueModel**: Advanced model considering order size advantages and market conditions

### Fee Structure
- **Maker Fees**: Lower fees for orders that provide liquidity
- **Taker Fees**: Higher fees for orders that take liquidity
- Configurable fee rates through the `FeeModel`

### Latency Simulation
- **Network Latency**: Simulates network transmission delays
- **Order Processing Latency**: Simulates exchange processing time
- Both use configurable latency models (e.g., Gaussian distribution)

## Architecture

### Core Components

#### OrderBook
- Maintains separate bid and ask price levels
- Tracks order count and total size at each level
- Supports FIFO order matching within price levels

#### OrderBookLevel
- Represents a single price level in the order book
- Contains list of orders at that price level
- Tracks total size and order count

#### QueueModel
- Abstract interface for queue position simulation
- Determines execution probability based on queue position
- Supports different market microstructure models

### Order Matching Logic

1. **Price-Time Priority**: Orders are matched by price first, then by time
2. **Queue Position**: Orders within the same price level are executed based on queue position
3. **Partial Fills**: Orders can be partially filled with remaining size staying in the book
4. **Market Impact**: Large orders may experience price impact through queue position effects

## Usage

### Basic Setup

```python
from gnomepy.backtest.exchanges.mbp import MBPSimulatedExchange
from gnomepy.backtest.fee import FeeModel
from gnomepy.backtest.latency import GaussianLatency
from gnomepy.backtest.queues import SimpleQueueModel

# Create components
fee_model = FeeModel(taker_fee=0.001, maker_fee=0.0005)
network_latency = GaussianLatency(mu=1000, sigma=100)
order_processing_latency = GaussianLatency(mu=500, sigma=50)
queue_model = SimpleQueueModel(base_execution_probability=0.8)

# Create exchange
exchange = MBPSimulatedExchange(
    fee_model=fee_model,
    network_latency=network_latency,
    order_processing_latency=order_processing_latency,
    queue_model=queue_model
)
```

### Submitting Orders

```python
from gnomepy import Order, OrderType, TimeInForce

# Create a limit order
order = Order(
    exchange_id=1,
    security_id=100,
    client_oid="order_001",
    price=10000,  # $10.00 (assuming 1e9 scale)
    size=1000,    # 1000 shares
    side="B",     # Buy
    order_type=OrderType.LIMIT,
    time_in_force=TimeInForce.GTC
)

# Submit order
execution_report = exchange.submit_order(order)
```

### Market Data Integration

The exchange can be updated with market data to provide more realistic pricing:

```python
# Update exchange with market data
exchange.update_market_data(market_data_record)

# Market orders will use this data for pricing when order book is empty
```

## Queue Models

### SimpleQueueModel
- Basic FIFO behavior
- Execution probability decays linearly with queue position
- Suitable for basic simulations

### RealisticQueueModel
- Considers order size advantages
- Exponential decay of execution probability
- Includes market noise simulation
- More suitable for realistic HFT simulations

## Supported Schemas

The exchange supports the following market data schemas:
- `MBP_10`: Market By Price with 10 levels
- `MBP_1`: Market By Price with 1 level
- `BBO_1M`: Best Bid/Offer with 1-minute aggregation
- `BBO_1S`: Best Bid/Offer with 1-second aggregation

## Example Scenarios

### Scenario 1: Basic Limit Order Matching
1. Submit buy limit order at $10.00
2. Submit sell limit order at $9.99
3. Orders match at $9.99 (seller's price)
4. Both orders are filled

### Scenario 2: Market Order with Queue Position
1. Multiple limit orders at same price level
2. Market order arrives
3. Queue position determines execution order
4. Some orders may be skipped due to queue position effects

### Scenario 3: IOC/FOK Orders
1. Submit IOC order with no matching liquidity
2. Order is immediately rejected
3. No order book impact

## Performance Considerations

- **Memory Usage**: Order books are maintained in memory for each exchange/security pair
- **Latency**: Order processing includes configurable latency simulation
- **Scalability**: Designed for single-threaded backtesting scenarios

## Integration with Backtesting Framework

The `MBPSimulatedExchange` integrates seamlessly with the gnomepy backtesting framework:

```python
from gnomepy.backtest.driver import Backtest

# Create backtest with exchange
backtest = Backtest(
    start_datetime=start_dt,
    end_datetime=end_dt,
    listing_ids=[1, 2, 3],
    schema_type=SchemaType.MBP_10,
    strategy=strategy,
    exchanges={1: exchange},  # Map exchange_id to exchange instance
    market_data_client=client
)
```

## Future Enhancements

- **Order Cancellation**: Full support for order cancellation
- **Market Data Feed**: Real-time market data integration
- **Risk Management**: Position limits and risk checks
- **Multi-threading**: Support for concurrent order processing
- **Market Impact Models**: More sophisticated price impact simulation 