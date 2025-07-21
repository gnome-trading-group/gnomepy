from gnomepy import SimulatedExchange, SchemaType, Order, OrderExecutionReport, FeeModel, LatencyModel, OrderType, TimeInForce, OrderStatus, ExecType
from gnomepy.backtest.queues import QueueModel
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time
import random


@dataclass
class OrderBookLevel:
    """Represents a price level in the order book"""
    price: int
    total_size: int
    order_count: int
    orders: List[Tuple[str, int]]  # (client_oid, size) pairs


@dataclass
class OrderBook:
    """Represents the order book for a security"""
    bids: Dict[int, OrderBookLevel]  # price -> level (sorted descending)
    asks: Dict[int, OrderBookLevel]  # price -> level (sorted ascending)
    
    def get_best_bid(self) -> Optional[int]:
        """Get the best bid price"""
        return max(self.bids.keys()) if self.bids else None
    
    def get_best_ask(self) -> Optional[int]:
        """Get the best ask price"""
        return min(self.asks.keys()) if self.asks else None
    
    def add_order(self, order: Order) -> None:
        """Add an order to the book"""
        price = order.price
        side = order.side
        client_oid = order.client_oid or f"internal_{time.time_ns()}"
        
        if side == "B":
            levels = self.bids
        else:
            levels = self.asks
            
        if price not in levels:
            levels[price] = OrderBookLevel(price, 0, 0, [])
            
        level = levels[price]
        level.total_size += order.size
        level.order_count += 1
        level.orders.append((client_oid, order.size))
    
    def remove_order(self, price: int, side: str, client_oid: str, size: int) -> bool:
        """Remove an order from the book"""
        levels = self.bids if side == "B" else self.asks
        
        if price not in levels:
            return False
            
        level = levels[price]
        
        # Find and remove the order
        for i, (oid, order_size) in enumerate(level.orders):
            if oid == client_oid and order_size == size:
                level.orders.pop(i)
                level.total_size -= size
                level.order_count -= 1
                
                # Remove empty level
                if level.total_size == 0:
                    del levels[price]
                    
                return True
                
        return False
    
    def get_matching_orders(self, order: Order, queue_model: QueueModel) -> List[Tuple[int, int, str]]:
        """Get orders that match against the incoming order with queue position consideration"""
        matches = []
        
        if order.side == "B":  # Buy order matches against asks
            for price in sorted(self.asks.keys()):
                if order.order_type == OrderType.LIMIT and price > order.price:
                    break
                    
                level = self.asks[price]
                remaining_size = order.size
                
                for i, (client_oid, size) in enumerate(level.orders):
                    if remaining_size <= 0:
                        break
                        
                    # Apply queue position logic
                    queue_position = queue_model.get_queue_position(size, level.total_size)
                    execution_probability = 0.9  # Base execution probability
                    
                    if queue_model.should_execute_at_position(queue_position, execution_probability):
                        match_size = min(remaining_size, size)
                        matches.append((price, match_size, client_oid))
                        remaining_size -= match_size
                    else:
                        # Skip this order due to queue position
                        continue
                        
                order.size = remaining_size
                if order.size == 0:
                    break
                    
        else:  # Sell order matches against bids
            for price in sorted(self.bids.keys(), reverse=True):
                if order.order_type == OrderType.LIMIT and price < order.price:
                    break
                    
                level = self.bids[price]
                remaining_size = order.size
                
                for i, (client_oid, size) in enumerate(level.orders):
                    if remaining_size <= 0:
                        break
                        
                    # Apply queue position logic
                    queue_position = queue_model.get_queue_position(size, level.total_size)
                    execution_probability = 0.9  # Base execution probability
                    
                    if queue_model.should_execute_at_position(queue_position, execution_probability):
                        match_size = min(remaining_size, size)
                        matches.append((price, match_size, client_oid))
                        remaining_size -= match_size
                    else:
                        # Skip this order due to queue position
                        continue
                        
                order.size = remaining_size
                if order.size == 0:
                    break
                    
        return matches


class MBPSimulatedExchange(SimulatedExchange):

    def __init__(
            self,
            fee_model: FeeModel,
            network_latency: LatencyModel,
            order_processing_latency: LatencyModel,
            queue_model: QueueModel,
    ):
        super().__init__(fee_model, network_latency, order_processing_latency)
        self.queue_model = queue_model
        self.order_books: Dict[Tuple[int, int], OrderBook] = {}  # (exchange_id, security_id) -> OrderBook
        self.open_orders: Dict[str, Order] = {}  # client_oid -> Order
        self.order_counter = 0
        self.market_data_state: Dict[Tuple[int, int], Dict] = {}  # Store latest market data

    def _get_or_create_order_book(self, exchange_id: int, security_id: int) -> OrderBook:
        """Get or create an order book for the given exchange/security pair"""
        key = (exchange_id, security_id)
        if key not in self.order_books:
            self.order_books[key] = OrderBook({}, {})
        return self.order_books[key]

    def _generate_client_oid(self) -> str:
        """Generate a unique client order ID"""
        self.order_counter += 1
        return f"client_{self.order_counter}_{time.time_ns()}"

    def _calculate_fee(self, price: int, size: int, is_maker: bool) -> int:
        """Calculate the fee for a trade"""
        fee_rate = self.fee_model.maker_fee if is_maker else self.fee_model.taker_fee
        return int(price * size * fee_rate)

    def _create_execution_report(
        self, 
        order: Order, 
        exec_type: ExecType, 
        order_status: OrderStatus,
        filled_qty: int = 0,
        filled_price: int = 0,
        leaves_qty: int = 0
    ) -> OrderExecutionReport:
        """Create an execution report"""
        return OrderExecutionReport(
            exchange_id=order.exchange_id,
            security_id=order.security_id,
            client_oid=order.client_oid,
            exec_type=exec_type,
            order_status=order_status,
            filled_qty=filled_qty,
            filled_price=filled_price,
            cumulative_qty=filled_qty,
            leaves_qty=leaves_qty,
            timestamp_event=time.time_ns(),
            timestamp_recv=time.time_ns()
        )

    def update_market_data(self, market_data) -> None:
        """Update the exchange's market data state"""
        key = (market_data.exchange_id, market_data.security_id)
        self.market_data_state[key] = {
            'timestamp': market_data.timestamp_recv,
            'data': market_data
        }

    def _get_market_price(self, exchange_id: int, security_id: int, side: str) -> Optional[int]:
        """Get the current market price for a security"""
        key = (exchange_id, security_id)
        if key in self.market_data_state:
            data = self.market_data_state[key]['data']
            if hasattr(data, 'levels'):
                if isinstance(data.levels, list):
                    # MBP10 - multiple levels
                    if side == "B":
                        return data.levels[0].ask_px if data.levels else None
                    else:
                        return data.levels[0].bid_px if data.levels else None
                else:
                    # MBP1 - single level
                    if side == "B":
                        return data.levels.ask_px
                    else:
                        return data.levels.bid_px
        return None

    def submit_order(self, order: Order) -> OrderExecutionReport:
        """Submit an order to the exchange"""
        # Generate client OID if not provided
        if order.client_oid is None:
            order.client_oid = self._generate_client_oid()
            
        order_book = self._get_or_create_order_book(order.exchange_id, order.security_id)
        
        # Handle different order types and time in force
        if order.order_type == OrderType.MARKET:
            return self._handle_market_order(order, order_book)
        elif order.order_type == OrderType.LIMIT:
            return self._handle_limit_order(order, order_book)
        else:
            # Reject unsupported order types
            return self._create_execution_report(
                order, ExecType.REJECT, OrderStatus.REJECTED
            )

    def _handle_market_order(self, order: Order, order_book: OrderBook) -> OrderExecutionReport:
        """Handle a market order"""
        original_size = order.size
        total_filled = 0
        weighted_price = 0
        is_maker = False  # Market orders are always takers
        
        # Get matching orders with queue position consideration
        matches = order_book.get_matching_orders(order, self.queue_model)
        
        if not matches:
            # No liquidity available
            if order.time_in_force == TimeInForce.IOC or order.time_in_force == TimeInForce.FOK:
                return self._create_execution_report(
                    order, ExecType.REJECT, OrderStatus.REJECTED
                )
            else:
                # Convert to limit order at market
                best_price = order_book.get_best_ask() if order.side == "B" else order_book.get_best_bid()
                if best_price is None:
                    # Try to get price from market data
                    best_price = self._get_market_price(order.exchange_id, order.security_id, order.side)
                    if best_price is None:
                        return self._create_execution_report(
                            order, ExecType.REJECT, OrderStatus.REJECTED
                        )
                order.price = best_price
                order.order_type = OrderType.LIMIT
                return self._handle_limit_order(order, order_book)
        
        # Process matches
        for match_price, match_size, match_client_oid in matches:
            total_filled += match_size
            weighted_price += match_price * match_size
            
            # Remove matched orders from book
            order_book.remove_order(match_price, "A" if order.side == "B" else "B", match_client_oid, match_size)
            
            # Update open order if it exists
            if match_client_oid in self.open_orders:
                matched_order = self.open_orders[match_client_oid]
                matched_order.size -= match_size
                if matched_order.size == 0:
                    del self.open_orders[match_client_oid]
        
        # Calculate fees
        total_fee = self._calculate_fee(weighted_price, total_filled, is_maker)
        
        # Determine execution type and status
        if total_filled == original_size:
            exec_type = ExecType.FILL
            order_status = OrderStatus.FILLED
        else:
            exec_type = ExecType.PARTIAL_FILL
            order_status = OrderStatus.PARTIALLY_FILLED
            
            # Add remaining size to book if not IOC/FOK
            if order.time_in_force != TimeInForce.IOC and order.time_in_force != TimeInForce.FOK:
                order.size = original_size - total_filled
                order_book.add_order(order)
                self.open_orders[order.client_oid] = order
        
        return self._create_execution_report(
            order, exec_type, order_status,
            filled_qty=total_filled,
            filled_price=weighted_price // total_filled if total_filled > 0 else 0,
            leaves_qty=original_size - total_filled
        )

    def _handle_limit_order(self, order: Order, order_book: OrderBook) -> OrderExecutionReport:
        """Handle a limit order"""
        original_size = order.size
        total_filled = 0
        weighted_price = 0
        
        # Check if order can be immediately filled
        matches = order_book.get_matching_orders(order, self.queue_model)
        
        if matches:
            # Order can be partially or fully filled
            is_maker = False  # Taking liquidity
            
            for match_price, match_size, match_client_oid in matches:
                total_filled += match_size
                weighted_price += match_price * match_size
                
                # Remove matched orders from book
                order_book.remove_order(match_price, "A" if order.side == "B" else "B", match_client_oid, match_size)
                
                # Update open order if it exists
                if match_client_oid in self.open_orders:
                    matched_order = self.open_orders[match_client_oid]
                    matched_order.size -= match_size
                    if matched_order.size == 0:
                        del self.open_orders[match_client_oid]
            
            # Calculate fees
            total_fee = self._calculate_fee(weighted_price, total_filled, is_maker)
            
            # Determine execution type and status
            if total_filled == original_size:
                exec_type = ExecType.FILL
                order_status = OrderStatus.FILLED
            else:
                exec_type = ExecType.PARTIAL_FILL
                order_status = OrderStatus.PARTIALLY_FILLED
                
                # Add remaining size to book
                order.size = original_size - total_filled
                order_book.add_order(order)
                self.open_orders[order.client_oid] = order
                
                # Send NEW execution report for the remaining order
                if order.time_in_force != TimeInForce.IOC and order.time_in_force != TimeInForce.FOK:
                    # This would typically be sent as a separate event
                    pass
        else:
            # Order cannot be filled immediately
            if order.time_in_force == TimeInForce.IOC or order.time_in_force == TimeInForce.FOK:
                # Reject IOC/FOK orders that cannot be filled
                return self._create_execution_report(
                    order, ExecType.REJECT, OrderStatus.REJECTED
                )
            else:
                # Add to order book
                order_book.add_order(order)
                self.open_orders[order.client_oid] = order
                
                exec_type = ExecType.NEW
                order_status = OrderStatus.NEW
                total_filled = 0
                weighted_price = 0
        
        return self._create_execution_report(
            order, exec_type, order_status,
            filled_qty=total_filled,
            filled_price=weighted_price // total_filled if total_filled > 0 else 0,
            leaves_qty=original_size - total_filled
        )

    def cancel_order(self, client_oid: str) -> OrderExecutionReport:
        """Cancel an open order"""
        if client_oid not in self.open_orders:
            return self._create_execution_report(
                Order(exchange_id=0, security_id=0, client_oid=client_oid, price=0, size=0, side="B", order_type=OrderType.LIMIT, time_in_force=TimeInForce.GTC),
                ExecType.CANCEL_REJECT, OrderStatus.REJECTED
            )
        
        order = self.open_orders[client_oid]
        order_book = self._get_or_create_order_book(order.exchange_id, order.security_id)
        
        # Remove from order book
        order_book.remove_order(order.price, order.side, client_oid, order.size)
        del self.open_orders[client_oid]
        
        return self._create_execution_report(
            order, ExecType.CANCEL, OrderStatus.CANCELED,
            leaves_qty=0
        )

    def get_supported_schemas(self) -> list[SchemaType]:
        return [SchemaType.MBP_10, SchemaType.MBP_1, SchemaType.BBO_1M, SchemaType.BBO_1S]
