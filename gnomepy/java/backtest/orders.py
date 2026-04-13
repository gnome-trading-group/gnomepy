from __future__ import annotations

from dataclasses import dataclass

from gnomepy.java.enums import ExecType, OrderStatus, Side
from gnomepy.java.statics import Scales


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
        dec = java_report.decoder
        return cls(
            client_oid=str(java_report.getClientOidCounter()),
            side=Side.NONE,
            exec_type=ExecType.from_java(dec.execType()),
            order_status=OrderStatus.from_java(dec.orderStatus()),
            filled_qty=int(dec.filledQty()),
            fill_price=int(dec.fillPrice()),
            cumulative_qty=int(dec.cumulativeQty()),
            leaves_qty=int(dec.leavesQty()),
            fee=int(dec.fee()) / Scales.PRICE,
            timestamp_event=int(dec.timestampEvent()),
            timestamp_recv=int(dec.timestampRecv()),
            exchange_id=int(dec.exchangeId()),
            security_id=int(dec.securityId()),
        )
