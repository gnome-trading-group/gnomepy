from __future__ import annotations

from dataclasses import dataclass

import jpype

from gnomepy.java.enums import ExecType, OrderStatus, RejectReason
from gnomepy.java.statics import Scales


@dataclass
class ExecutionReport:
    """Python-friendly representation of a backtest execution report."""

    client_oid: str
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
    reject_reason: RejectReason | None = None

    @classmethod
    def _from_java(cls, java_report) -> ExecutionReport:
        """Copies a Java report, reading any field the sender left as its SBE null as 0.

        Synthetic OMS rejects leave the fill fields and fee null; read raw, the null fee
        alone would be about -$9.2 billion.
        """
        dec = java_report.decoder
        nulls = jpype.JClass("group.gnometrading.schemas.OrderExecutionReportDecoder")
        return cls(
            client_oid=str(java_report.getClientOidCounter()),
            exec_type=ExecType.from_java(dec.execType()),
            order_status=OrderStatus.from_java(dec.orderStatus()),
            filled_qty=_or_zero(dec.filledQty(), nulls.filledQtyNullValue()),
            fill_price=_or_zero(dec.fillPrice(), nulls.fillPriceNullValue()),
            cumulative_qty=_or_zero(dec.cumulativeQty(), nulls.cumulativeQtyNullValue()),
            leaves_qty=_or_zero(dec.leavesQty(), nulls.leavesQtyNullValue()),
            fee=_or_zero(dec.fee(), nulls.feeNullValue()) / Scales.PRICE,
            timestamp_event=_or_zero(dec.timestampEvent(), nulls.timestampEventNullValue()),
            timestamp_recv=_or_zero(dec.timestampRecv(), nulls.timestampRecvNullValue()),
            exchange_id=int(dec.exchangeId()),
            security_id=int(dec.securityId()),
            reject_reason=RejectReason.from_java(dec.rejectReason()),
        )


def _or_zero(value, null_value) -> int:
    value = int(value)
    return 0 if value == int(null_value) else value
