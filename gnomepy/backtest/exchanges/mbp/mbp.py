from gnomepy.backtest.exchanges.base import SimulatedExchange
from gnomepy.backtest.exchanges.mbp.mbp_book import MBPBook
from gnomepy.backtest.exchanges.mbp.types import LocalOrder
from gnomepy.backtest.fee import FeeModel
from gnomepy.backtest.latency import LatencyModel
from gnomepy.backtest.queues.base import QueueModel
from gnomepy.data.types import SchemaType, Order, OrderExecutionReport, OrderType, \
    TimeInForce, OrderStatus, ExecType, SchemaBase, MBP10, MBP1
import time


def _create_execution_report(
    client_oid: str,
    exec_type: ExecType,
    order_status: OrderStatus,
    filled_qty: int = 0,
    filled_price: int = 0,
    cumulative_qty: int = 0,
    leaves_qty: int = 0,
    fee: float = 0.0
) -> OrderExecutionReport:
    return OrderExecutionReport(
        client_oid=client_oid,
        exec_type=exec_type,
        order_status=order_status,
        filled_qty=filled_qty,
        filled_price=filled_price,
        cumulative_qty=cumulative_qty,
        leaves_qty=leaves_qty,
        exchange_id=-1,
        security_id=-1,
        timestamp_event=-1,
        timestamp_recv=-1,
        fee=fee,
    )

class MBPSimulatedExchange(SimulatedExchange):

    def __init__(
            self,
            fee_model: FeeModel,
            network_latency: LatencyModel,
            order_processing_latency: LatencyModel,
            queue_model: QueueModel,
    ):
        super().__init__(fee_model, network_latency, order_processing_latency)
        self.order_book = MBPBook(queue_model)
        self.open_orders: dict[str, Order] = {}  # client_oid -> Order
        self.order_counter = 0

    def _generate_client_oid(self) -> str:
        """Generate a unique client order ID"""
        self.order_counter += 1
        return f"client_{self.order_counter}_{time.time_ns()}"

    def on_market_data(self, data: SchemaBase) -> list[OrderExecutionReport]:
        if not isinstance(data, (MBP10, MBP1)):
            raise ValueError(f"Invalid market data type: {type(data)}")

        if data.action in ("Add", "Cancel", "Modify"): # "A", "C", "M"
            fills = self.order_book.on_market_update(data.levels)
        elif data.action == "Trade":
            fills = self.order_book.on_trade(data)
        else:
            fills = []
        return [self._map_execution_report(fill[0], fill[1]) for fill in fills]

    def _map_execution_report(self, local_order: LocalOrder, filled_qty: int) -> OrderExecutionReport:
        total_price = filled_qty * local_order.order.price
        total_fee = self.fee_model.calculate_fee(total_price, is_maker=True)
        filled_price = total_price // filled_qty

        return _create_execution_report(
            client_oid=local_order.order.client_oid,
            exec_type=ExecType.TRADE,
            order_status=OrderStatus.FILLED if local_order.remaining == 0 else OrderStatus.PARTIALLY_FILLED,
            filled_qty=filled_qty,
            cumulative_qty=local_order.order.size - local_order.remaining,
            filled_price=filled_price,
            leaves_qty=local_order.remaining,
            fee=total_fee,
        )

    def submit_order(self, order: Order) -> list[OrderExecutionReport]:
        if order.client_oid is None:
            order.client_oid = self._generate_client_oid()
            
        if order.order_type == OrderType.MARKET:
            return self._handle_market_order(order)
        elif order.order_type == OrderType.LIMIT:
            return self._handle_limit_order(order)
        else:
            raise ValueError(f"Unexpected order type: {order.order_type}")

    def _handle_market_order(self, order: Order) -> list[OrderExecutionReport]:
        matches = self.order_book.get_matching_orders(order)
        if not matches:
            raise ValueError("Not enough liquidity to fulfill order")
            # if order.time_in_force == TimeInForce.IOC or order.time_in_force == TimeInForce.FOK:
            #     return self._create_execution_report(order.client_oid, ExecType.REJECT, OrderStatus.REJECTED)
            # else:
            #     best_price = self.order_book.get_best_ask() if order.side == "B" else self.order_book.get_best_bid()
            #     if best_price is None:
            #         raise ValueError("Bad limit order book state")
            #     order.order_type = OrderType.LIMIT
            #     order.price = best_price
            #     return self._handle_limit_order(order)

        total_filled = 0
        total_price = 0
        for match in matches:
            total_filled += match.size
            total_price += match.price * match.size

        total_fee = self.fee_model.calculate_fee(total_price, is_maker=False)
        # total_price += total_fee if order.side == 'B' else -total_fee

        if total_filled == order.size:
            exec_type = ExecType.TRADE
            order_status = OrderStatus.FILLED
        else:
            raise ValueError("Not enough liquidity")
        
        return [
            _create_execution_report(
                order.client_oid, exec_type, order_status,
                filled_qty=total_filled,
                cumulative_qty=total_filled,
                filled_price=total_price // total_filled,
                leaves_qty=order.size - total_filled,
                fee=total_fee,
            )
        ]

    def _handle_limit_order(self, order: Order) -> list[OrderExecutionReport]:
        matches = self.order_book.get_matching_orders(order)

        new_order_report = _create_execution_report(
            client_oid=order.client_oid,
            exec_type=ExecType.NEW,
            order_status=OrderStatus.NEW,
            leaves_qty=order.size
        )
        
        if matches:
            total_filled = 0
            total_price = 0
            for match in matches:
                total_filled += match.size
                total_price += match.price * match.size

            total_fee = self.fee_model.calculate_fee(total_price, False)
            # total_price += total_fee if order.side == 'B' else -total_fee

            if total_filled == order.size:
                return [
                    _create_execution_report(
                        order.client_oid, ExecType.TRADE, OrderStatus.FILLED,
                        filled_qty=total_filled,
                        cumulative_qty=total_filled,
                        filled_price=total_price // total_filled,
                        leaves_qty=0,
                        fee=total_fee,
                    )
                ]
            else:
                partial_fill_report = _create_execution_report(
                    order.client_oid, ExecType.TRADE, OrderStatus.PARTIALLY_FILLED,
                    filled_qty=total_filled,
                    cumulative_qty=total_filled,
                    filled_price=total_price // total_filled,
                    leaves_qty=order.size - total_filled,
                    fee=total_fee,
                )
                if order.time_in_force == TimeInForce.FOK:
                    return [
                        _create_execution_report(order.client_oid, ExecType.REJECTED, OrderStatus.REJECTED)
                    ]
                elif order.time_in_force == TimeInForce.IOC:
                    return [
                        partial_fill_report,
                        _create_execution_report(
                            order.client_oid, ExecType.CANCELED, OrderStatus.PARTIALLY_FILLED,
                            cumulative_qty=total_filled,
                            leaves_qty=0,
                            fee=total_fee,
                        )
                    ]
                else:
                    remaining = order.size - total_filled
                    self.order_book.add_local_order(order, remaining)
                    return [
                        new_order_report,
                        partial_fill_report,
                    ]
        else:
            if order.time_in_force == TimeInForce.IOC or order.time_in_force == TimeInForce.FOK:
                return [
                    _create_execution_report(
                        client_oid=order.client_oid,
                        exec_type=ExecType.REJECTED,
                        order_status=OrderStatus.REJECTED,
                    )
                ]
            else:
                self.order_book.add_local_order(order)
                return [
                    _create_execution_report(
                        client_oid=order.client_oid,
                        exec_type=ExecType.NEW,
                        order_status=OrderStatus.NEW,
                        leaves_qty=order.size
                    )
                ]

    def cancel_order(self, client_oid: str) -> list[OrderExecutionReport]:
        if self.order_book.cancel_order(client_oid):
            return [_create_execution_report(client_oid, ExecType.CANCELED, OrderStatus.CANCELED)]
        return [_create_execution_report(client_oid, ExecType.REJECTED, OrderStatus.REJECTED)]

    def get_supported_schemas(self) -> list[SchemaType]:
        return [SchemaType.MBP_10, SchemaType.MBP_1]
