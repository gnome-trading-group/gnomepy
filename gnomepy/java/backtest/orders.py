from __future__ import annotations

from dataclasses import dataclass

from gnomepy.java.enums import ExecType, OrderStatus, Side


@dataclass
class ExecutionReport:
    """Python-friendly representation of a backtest execution report."""

    client_oid: str
    side: Side
    exec_type: ExecType
    order_status: OrderStatus
    filled_qty: int
    fill_price: int
    cumulative_qty: int
    leaves_qty: int
    fee: float
    timestamp_event: int
    timestamp_recv: int
    exchange_id: int
    security_id: int

    @classmethod
    def _from_java(cls, java_report) -> ExecutionReport:
        return cls(
            client_oid=str(java_report.clientOid),
            side=Side.from_java(java_report.side),
            exec_type=ExecType.from_java(java_report.execType),
            order_status=OrderStatus.from_java(java_report.orderStatus),
            filled_qty=int(java_report.filledQty),
            fill_price=int(java_report.fillPrice),
            cumulative_qty=int(java_report.cumulativeQty),
            leaves_qty=int(java_report.leavesQty),
            fee=float(java_report.fee),
            timestamp_event=int(java_report.timestampEvent),
            timestamp_recv=int(java_report.timestampRecv),
            exchange_id=int(java_report.exchangeId),
            security_id=int(java_report.securityId),
        )
