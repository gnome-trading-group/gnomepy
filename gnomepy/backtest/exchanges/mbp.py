from gnomepy.backtest.exchanges import SimulatedExchange
from gnomepy.backtest.fee import FeeModel
from gnomepy.backtest.latency import LatencyModel
from gnomepy.backtest.queues import QueueModel
from gnomepy.data.types import SchemaType, Order, OrderExecutionReport, OrderType, \
    TimeInForce, OrderStatus, ExecType, SchemaBase, BBO1S, MBP10, MBP1, BBO1M, BidAskPair
from dataclasses import dataclass
import time


@dataclass
class OrderBookLevel:
    price: int
    total_size: int
    order_count: int

@dataclass
class OrderMatch:
    price: int
    size: int

@dataclass
class MBPBook:
    levels: list[BidAskPair]
    
    def get_best_bid(self) -> int | None:
        return self.levels[0].bid_px if self.levels else None
    
    def get_best_ask(self) -> int | None:
        return self.levels[0].ask_px if self.levels else None

    def update_levels(self, levels: list[BidAskPair]):
        self.levels = levels
    
    def add_local_order(self, order: Order) -> None:
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
    
    def get_matching_orders(self, order: Order) -> list[OrderMatch]:
        matches = []

        remaining_size = order.size
        comparison = lambda _price_level: \
            (_price_level.ask_px > order.price) if order.side == "B" else (_price_level.bid_px < order.price)
        
        for price_level in self.levels:
            if order.order_type == OrderType.LIMIT and comparison(price_level):
                break

            if remaining_size == 0:
                break

            if order.side == "B":
                match_size = min(remaining_size, price_level.ask_sz)
                match_price = price_level.ask_px
            else:
                match_size = min(remaining_size, price_level.bid_sz)
                match_price = price_level.bid_px

            remaining_size -= match_size
            matches.append(OrderMatch(price=match_price, size=match_size))

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
        self.order_book = MBPBook(levels=[])
        self.open_orders: dict[str, Order] = {}  # client_oid -> Order
        self.order_counter = 0

    def _generate_client_oid(self) -> str:
        """Generate a unique client order ID"""
        self.order_counter += 1
        return f"client_{self.order_counter}_{time.time_ns()}"

    def _create_execution_report(
        self, 
        client_oid: str,
        exec_type: ExecType, 
        order_status: OrderStatus,
        filled_qty: int = 0,
        filled_price: int = 0,
        leaves_qty: int = 0
    ) -> OrderExecutionReport:
        return OrderExecutionReport(
            client_oid=client_oid,
            exec_type=exec_type,
            order_status=order_status,
            filled_qty=filled_qty,
            filled_price=filled_price,
            cumulative_qty=filled_qty,
            leaves_qty=leaves_qty,

            exchange_id=-1,
            security_id=-1,
            timestamp_event=-1,
            timestamp_recv=-1,
        )

    def on_market_data(self, data: SchemaBase):
        if isinstance(data, (MBP10, MBP1)):
            if data.action in ("Add", "Cancel", "Modify", "Clear"): # "A", "C", "M", "W"
                self.order_book.update_levels(data.levels)
        elif isinstance(data, (BBO1S, BBO1M)):
            self.order_book.update_levels(data.levels)
        else:
            raise ValueError(f"Invalid market data type: {type(data)}")

    def submit_order(self, order: Order) -> OrderExecutionReport:
        if order.client_oid is None:
            order.client_oid = self._generate_client_oid()
            
        if order.order_type == OrderType.MARKET:
            return self._handle_market_order(order)
        elif order.order_type == OrderType.LIMIT:
            return self._handle_limit_order(order)
        else:
            raise ValueError(f"Unexpected order type: {order.order_type}")

    def _handle_market_order(self, order: Order) -> OrderExecutionReport:
        matches = self.order_book.get_matching_orders(order)
        if not matches:
            if order.time_in_force == TimeInForce.IOC or order.time_in_force == TimeInForce.FOK:
                return self._create_execution_report(order.client_oid, ExecType.REJECT, OrderStatus.REJECTED)
            else:
                best_price = self.order_book.get_best_ask() if order.side == "B" else self.order_book.get_best_bid()
                if best_price is None:
                    raise ValueError("Bad limit order book state")
                order.order_type = OrderType.LIMIT
                order.price = best_price
                return self._handle_limit_order(order)

        total_filled = 0
        total_price = 0
        for match in matches:
            total_filled += match.size
            total_price += match.price * match.size

        total_fee = self.fee_model.calculate_fee(total_price, False)
        total_price += total_fee
        
        if total_filled == order.size:
            exec_type = ExecType.FILL
            order_status = OrderStatus.FILLED
        else:
            raise ValueError("Not enough liquidity")
        
        return self._create_execution_report(
            order.client_oid, exec_type, order_status,
            filled_qty=total_filled,
            filled_price=total_price // total_filled if total_filled > 0 else 0,
            leaves_qty=order.size - total_filled
        )

    def _handle_limit_order(self, order: Order, order_book: MBPBook) -> OrderExecutionReport:
        raise NotImplementedError
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
        raise NotImplementedError
        if client_oid not in self.open_orders:
            return self._create_execution_report(client_oid, ExecType.CANCEL_REJECT, OrderStatus.REJECTED)
        
        order = self.open_orders[client_oid]

        order_book.remove_order(order.price, order.side, client_oid, order.size)
        del self.open_orders[client_oid]
        
        return self._create_execution_report(client_oid, ExecType.CANCEL, OrderStatus.CANCELED, leaves_qty=0)

    def get_supported_schemas(self) -> list[SchemaType]:
        return [SchemaType.MBP_10, SchemaType.MBP_1, SchemaType.BBO_1M, SchemaType.BBO_1S]
