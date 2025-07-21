#!/usr/bin/env python3
"""
Example script demonstrating the use of MBPSimulatedExchange.

This script shows how to create and use the MBPSimulatedExchange with different
order types, time in force values, and queue models.
"""

import time
from gnomepy import Order, OrderType, TimeInForce
from gnomepy.backtest.exchanges.mbp import MBPSimulatedExchange
from gnomepy.backtest.fee import FeeModel
from gnomepy.backtest.latency import GaussianLatency
from gnomepy.backtest.queues import SimpleQueueModel, RealisticQueueModel


def create_exchange(queue_model_type: str = "simple") -> MBPSimulatedExchange:
    """Create an MBPSimulatedExchange with the specified configuration"""
    
    # Create fee model (0.1% taker fee, 0.05% maker fee)
    fee_model = FeeModel(taker_fee=0.001, maker_fee=0.0005)
    
    # Create latency models (in nanoseconds)
    network_latency = GaussianLatency(mu=1000, sigma=100)  # 1 microsecond ± 100ns
    order_processing_latency = GaussianLatency(mu=500, sigma=50)  # 500ns ± 50ns
    
    # Create queue model
    if queue_model_type == "realistic":
        queue_model = RealisticQueueModel(base_execution_probability=0.8, size_advantage_factor=0.2)
    else:
        queue_model = SimpleQueueModel(base_execution_probability=0.8)
    
    return MBPSimulatedExchange(
        fee_model=fee_model,
        network_latency=network_latency,
        order_processing_latency=order_processing_latency,
        queue_model=queue_model
    )


def example_limit_orders():
    """Example of limit order execution"""
    print("=== Limit Order Example ===")
    
    exchange = create_exchange("simple")
    
    # Create a buy limit order
    buy_order = Order(
        exchange_id=1,
        security_id=100,
        client_oid="buy_001",
        price=10000,  # $10.00 (assuming 1e9 scale)
        size=1000,    # 1000 shares
        side="B",     # Buy
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC
    )
    
    # Submit the buy order
    print(f"Submitting buy order: {buy_order.client_oid}")
    report = exchange.submit_order(buy_order)
    print(f"Execution report: {report.exec_type}, Status: {report.order_status}")
    print(f"Filled: {report.filled_qty}, Leaves: {report.leaves_qty}")
    
    # Create a sell limit order that should match
    sell_order = Order(
        exchange_id=1,
        security_id=100,
        client_oid="sell_001",
        price=9990,   # $9.99 - should match against buy order
        size=500,     # 500 shares
        side="A",     # Sell
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC
    )
    
    # Submit the sell order
    print(f"\nSubmitting sell order: {sell_order.client_oid}")
    report = exchange.submit_order(sell_order)
    print(f"Execution report: {report.exec_type}, Status: {report.order_status}")
    print(f"Filled: {report.filled_qty}, Leaves: {report.leaves_qty}")
    print(f"Fill price: {report.filled_price / 1e9:.6f}")  # Convert to dollars


def example_market_orders():
    """Example of market order execution"""
    print("\n=== Market Order Example ===")
    
    exchange = create_exchange("realistic")
    
    # First, add some liquidity to the book
    liquidity_orders = [
        Order(1, 100, "liquidity_ask_1", 10010, 1000, "A", OrderType.LIMIT, TimeInForce.GTC),
        Order(1, 100, "liquidity_ask_2", 10020, 1000, "A", OrderType.LIMIT, TimeInForce.GTC),
        Order(1, 100, "liquidity_bid_1", 9990, 1000, "B", OrderType.LIMIT, TimeInForce.GTC),
        Order(1, 100, "liquidity_bid_2", 9980, 1000, "B", OrderType.LIMIT, TimeInForce.GTC),
    ]
    
    for order in liquidity_orders:
        exchange.submit_order(order)
    
    # Create a market buy order
    market_buy = Order(
        exchange_id=1,
        security_id=100,
        client_oid="market_buy_001",
        price=0,      # Price ignored for market orders
        size=1500,    # 1500 shares
        side="B",     # Buy
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.IOC
    )
    
    print(f"Submitting market buy order: {market_buy.client_oid}")
    report = exchange.submit_order(market_buy)
    print(f"Execution report: {report.exec_type}, Status: {report.order_status}")
    print(f"Filled: {report.filled_qty}, Leaves: {report.leaves_qty}")
    print(f"Fill price: {report.filled_price / 1e9:.6f}")


def example_time_in_force():
    """Example of different time in force behaviors"""
    print("\n=== Time In Force Example ===")
    
    exchange = create_exchange("simple")
    
    # Test IOC (Immediate or Cancel) order with no liquidity
    ioc_order = Order(
        exchange_id=1,
        security_id=200,
        client_oid="ioc_test",
        price=10000,
        size=1000,
        side="B",
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.IOC
    )
    
    print(f"Submitting IOC order with no liquidity: {ioc_order.client_oid}")
    report = exchange.submit_order(ioc_order)
    print(f"Execution report: {report.exec_type}, Status: {report.order_status}")
    
    # Test FOK (Fill or Kill) order
    fok_order = Order(
        exchange_id=1,
        security_id=200,
        client_oid="fok_test",
        price=10000,
        size=1000,
        side="B",
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.FOK
    )
    
    print(f"\nSubmitting FOK order with no liquidity: {fok_order.client_oid}")
    report = exchange.submit_order(fok_order)
    print(f"Execution report: {report.exec_type}, Status: {report.order_status}")


def example_queue_models():
    """Example comparing different queue models"""
    print("\n=== Queue Model Comparison ===")
    
    # Test with simple queue model
    simple_exchange = create_exchange("simple")
    
    # Add some orders to create a queue
    for i in range(5):
        order = Order(
            exchange_id=1,
            security_id=300,
            client_oid=f"queue_test_{i}",
            price=10000,
            size=100,
            side="B",
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.GTC
        )
        simple_exchange.submit_order(order)
    
    # Test with realistic queue model
    realistic_exchange = create_exchange("realistic")
    
    # Add the same orders
    for i in range(5):
        order = Order(
            exchange_id=1,
            security_id=300,
            client_oid=f"queue_test_{i}",
            price=10000,
            size=100,
            side="B",
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.GTC
        )
        realistic_exchange.submit_order(order)
    
    print("Queue models created with different execution probabilities")


def main():
    """Run all examples"""
    print("MBPSimulatedExchange Examples")
    print("=" * 50)
    
    example_limit_orders()
    example_market_orders()
    example_time_in_force()
    example_queue_models()
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    main() 