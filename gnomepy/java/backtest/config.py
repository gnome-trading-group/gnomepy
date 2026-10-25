from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Union

import jpype

from gnomepy.java.enums import RecordType


@dataclass
class RegistryFeeConfig:
    """Each listing pays what its venue charged at the time, from the registry's fee schedules."""

    def _to_java(self):
        return jpype.JClass("group.gnometrading.simulation.config.FeeModelConfig$Registry")()


@dataclass
class StaticFeeConfig:
    taker_fee: float = 0.0
    maker_fee: float = 0.0

    def _to_java(self):
        cls = jpype.JClass("group.gnometrading.simulation.config.FeeModelConfig$Static")
        obj = cls()
        obj.takerFee = float(self.taker_fee)
        obj.makerFee = float(self.maker_fee)
        return obj


@dataclass
class ParametricFeeConfig:
    taker_fee_rate: float = 0.07
    maker_fee_rate: float = 0.0

    def _to_java(self):
        cls = jpype.JClass("group.gnometrading.simulation.config.FeeModelConfig$Parametric")
        obj = cls()
        obj.takerFeeRate = float(self.taker_fee_rate)
        obj.makerFeeRate = float(self.maker_fee_rate)
        return obj


@dataclass
class StaticLatencyConfig:
    latency_nanos: int = 0

    def _to_java(self):
        cls = jpype.JClass("group.gnometrading.simulation.config.LatencyConfig$Static")
        obj = cls()
        obj.latencyNanos = jpype.JLong(self.latency_nanos)
        return obj


@dataclass
class LogNormalLatencyConfig:
    """A floor plus a log-normal tail, set by its floor, median and 99th percentile in nanoseconds.

    The defaults come from recorded Polymarket and Kalshi market data with the
    venues' publish batching taken out. ``seed`` pins this model's draws. Left as
    ``None``, the model takes its own stream derived from ``BacktestConfig.seed``,
    the listing and its role.
    """

    floor_nanos: int = 5_000_000
    median_nanos: int = 12_000_000
    p99_nanos: int = 200_000_000
    seed: int | None = None

    def _to_java(self):
        cls = jpype.JClass("group.gnometrading.simulation.config.LatencyConfig$LogNormal")
        obj = cls()
        obj.floorNanos = jpype.JLong(self.floor_nanos)
        obj.medianNanos = jpype.JLong(self.median_nanos)
        obj.p99Nanos = jpype.JLong(self.p99_nanos)
        if self.seed is not None:
            obj.seed = jpype.JObject(self.seed, jpype.JClass("java.lang.Long"))
        return obj


@dataclass
class RecordedLatencyConfig:
    """Market data arrives at the receive time recorded with each record.

    ``fallback`` is used only for a record with no usable receive time.
    """

    fallback: Union[StaticLatencyConfig, LogNormalLatencyConfig] = field(
        default_factory=lambda: StaticLatencyConfig(latency_nanos=50_000_000)
    )

    def _to_java(self):
        cls = jpype.JClass("group.gnometrading.simulation.config.LatencyConfig$Recorded")
        obj = cls()
        obj.fallback = self.fallback._to_java()
        return obj


@dataclass
class MakerTakerLatencyConfig:
    base_nanos: int = 0
    taker_delay_nanos: int = 0
    maker_delay_nanos: int = 0

    def _to_java(self):
        cls = jpype.JClass("group.gnometrading.simulation.config.LatencyConfig$MakerTaker")
        obj = cls()
        obj.baseNanos = jpype.JLong(self.base_nanos)
        obj.takerDelayNanos = jpype.JLong(self.taker_delay_nanos)
        obj.makerDelayNanos = jpype.JLong(self.maker_delay_nanos)
        return obj


@dataclass
class OptimisticQueueConfig:
    def _to_java(self):
        return jpype.JClass("group.gnometrading.simulation.config.QueueModelConfig$Optimistic")()


@dataclass
class RiskAverseQueueConfig:
    def _to_java(self):
        return jpype.JClass("group.gnometrading.simulation.config.QueueModelConfig$RiskAverse")()


@dataclass
class ProbabilisticQueueConfig:
    cancel_ahead_probability: float = 0.5

    def _to_java(self):
        cls = jpype.JClass("group.gnometrading.simulation.config.QueueModelConfig$Probabilistic")
        obj = cls()
        obj.cancelAheadProbability = float(self.cancel_ahead_probability)
        return obj


@dataclass
class SimulationOverrides:
    """Blocks of how a venue behaves in simulation, in place of its exchange's registry parameters.

    Every field left ``None`` keeps the layer below: the exchange's parameters recorded in the registry, or for a
    listing, its exchange's overrides. A block is replaced whole. Fees default to the registry's fee schedules.
    """

    fee_model: Union[RegistryFeeConfig, StaticFeeConfig, ParametricFeeConfig, None] = None
    # When market data reaches the strategy.
    market_data_latency: Union[RecordedLatencyConfig, StaticLatencyConfig, LogNormalLatencyConfig, None] = None
    # Our order connection: orders out and reports back.
    network_latency: Union[StaticLatencyConfig, LogNormalLatencyConfig, MakerTakerLatencyConfig, None] = None
    order_processing_latency: Union[StaticLatencyConfig, LogNormalLatencyConfig, MakerTakerLatencyConfig, None] = None
    queue_model: Union[OptimisticQueueConfig, RiskAverseQueueConfig, ProbabilisticQueueConfig, None] = None
    # What the venue does when our order would trade with our own resting order:
    # "CANCEL_INCOMING" (Kalshi taker_at_cross) or "CANCEL_RESTING" (Kalshi maker, which our Kalshi gateway sends).
    self_trade_prevention: str | None = None

    def _to_java(self):
        obj = jpype.JClass("group.gnometrading.simulation.config.SimulationOverrides")()
        for name, attr in (
            ("fee_model", "feeModel"),
            ("market_data_latency", "marketDataLatency"),
            ("network_latency", "networkLatency"),
            ("order_processing_latency", "orderProcessingLatency"),
            ("queue_model", "queueModel"),
        ):
            block = getattr(self, name)
            if block is not None:
                setattr(obj, attr, block._to_java())
        if self.self_trade_prevention is not None:
            obj.selfTradePrevention = jpype.JClass(
                "group.gnometrading.simulation.book.SelfTradePrevention"
            ).valueOf(self.self_trade_prevention)
        return obj


@dataclass
class ListingSimConfig:
    """A listing the backtest trades, with any blocks it simulates with in place of its exchange's."""

    listing_id: int
    overrides: SimulationOverrides | None = None

    def _to_java(self):
        cls = jpype.JClass("group.gnometrading.backtest.config.ListingSimConfig")
        obj = cls()
        obj.listingId = jpype.JInt(self.listing_id)
        if self.overrides is not None:
            obj.overrides = self.overrides._to_java()
        return obj


@dataclass
class StreamConfig:
    """A stream the strategy reads besides its listings' books (a vendor feed, say), and which of its records to replay."""

    stream_id: int
    record_type: RecordType

    def _to_java(self):
        cls = jpype.JClass("group.gnometrading.backtest.config.StreamConfig")
        obj = cls()
        obj.streamId = jpype.JInt(self.stream_id)
        obj.recordType = self.record_type.value
        return obj


@dataclass
class StrategyConfig:
    """Strategy class name and constructor args for YAML-based strategy resolution."""

    class_name: str
    args: dict | None = None

    def _to_java(self):
        cls = jpype.JClass("group.gnometrading.backtest.config.StrategyConfig")
        HashMap = jpype.JClass("java.util.HashMap")
        obj = cls()
        obj.className = str(self.class_name)
        if self.args:
            m = HashMap()
            for k, v in self.args.items():
                m.put(str(k), v)
            obj.args = m
        return obj


@dataclass
class PolicyConfig:
    """One OMS risk policy.

    ``type`` is a RiskPolicyType name. Targets work as in the registry: 0 means
    every strategy, exchange or listing, so a policy with none applies to
    everything; a policy names an exchange or a listing, not both. Money limits (maxOrderNotional, maxCapital, maxLoss) are in
    price units (1e9 = $1); size limits (maxOrderSize, maxPosition) are in size
    units (1e6 = 1 unit); maxOpenOrders and maxRejects are counts.
    """

    type: str
    params: dict[str, object] = field(default_factory=dict)
    strategy_id: int = 0
    exchange_id: int = 0
    listing_id: int = 0

    def _to_java(self):
        obj = jpype.JClass("group.gnometrading.backtest.config.RiskConfig$PolicyConfig")()
        obj.type = str(self.type)
        obj.strategyId = jpype.JInt(self.strategy_id)
        obj.exchangeId = jpype.JInt(self.exchange_id)
        obj.listingId = jpype.JInt(self.listing_id)
        params = jpype.JClass("java.util.HashMap")()
        for k, v in (self.params or {}).items():
            params.put(str(k), v)
        obj.params = params
        return obj


@dataclass
class RiskConfig:
    """OMS risk policy configuration.

    Example::

        RiskConfig(policies=[
            PolicyConfig("MAX_ORDER_NOTIONAL", {"maxOrderNotional": 100_000 * 1_000_000_000}),
            PolicyConfig("MAX_OPEN_ORDERS", {"maxOpenOrders": 3}, listing_id=7122),
        ])

    With ``from_registry``, the strategy's live policies for
    ``BacktestConfig.strategy_id`` are loaded too, without kill switches.

    KILL_SWITCH is rejected: it stops all trading, which has no meaning in a backtest.
    """

    policies: list[PolicyConfig] = field(default_factory=list)
    from_registry: bool = False

    def _to_java(self):
        obj = jpype.JClass("group.gnometrading.backtest.config.RiskConfig")()
        policies = jpype.JClass("java.util.ArrayList")()
        for policy in self.policies:
            policies.add(policy._to_java())
        obj.policies = policies
        obj.fromRegistry = self.from_registry
        return obj


@dataclass
class BacktestConfig:
    """Full backtest configuration.

    Usage::

        config = BacktestConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 2),
            listings=[ListingSimConfig(listing_id=1)],
            exchanges={"KALSHI": SimulationOverrides(queue_model=OptimisticQueueConfig())},
        )
        results = run_backtest(config, strategy=MyStrategy())
    """

    start_date: date | datetime
    end_date: date | datetime
    listings: list[ListingSimConfig]
    # Streams the strategy reads besides its listings' books (a vendor feed, say). Their records reach the strategy at
    # their recorded receive time; no exchange simulates them. The listings' books replay as MBP-10, the record the
    # simulated exchanges match against.
    streams: list[StreamConfig] = field(default_factory=list)
    # Overrides for every listing on an exchange, by its code (e.g. KALSHI); listings simulate with their exchange's
    # registry parameters otherwise.
    exchanges: dict[str, SimulationOverrides] = field(default_factory=dict)
    # The time the run reads the registry's simulation parameters and fees as known at (ISO-8601); None means when
    # the run starts. Rerunning with the time a run recorded simulates it identically.
    simulation_as_of: str | None = None
    strategy: StrategyConfig | None = None
    strategy_id: int = 0
    risk: RiskConfig = field(default_factory=RiskConfig)
    record: bool = True
    record_depth: int = 1
    measure_processing_time: bool = False
    seed: int | None = None

    def _to_java(self):
        cls = jpype.JClass("group.gnometrading.backtest.config.BacktestConfig")
        ArrayList = jpype.JClass("java.util.ArrayList")
        LinkedHashMap = jpype.JClass("java.util.LinkedHashMap")
        LocalDateTime = jpype.JClass("java.time.LocalDateTime")

        start = self.start_date if isinstance(self.start_date, datetime) else datetime(
            self.start_date.year, self.start_date.month, self.start_date.day
        )
        end = self.end_date if isinstance(self.end_date, datetime) else datetime(
            self.end_date.year, self.end_date.month, self.end_date.day
        )

        obj = cls()
        obj.startDate = LocalDateTime.of(
            start.year, start.month, start.day, start.hour, start.minute, start.second
        )
        obj.endDate = LocalDateTime.of(
            end.year, end.month, end.day, end.hour, end.minute, end.second
        )

        listings_list = ArrayList()
        for listing in self.listings:
            listings_list.add(listing._to_java())
        obj.listings = listings_list
        streams_list = ArrayList()
        for stream in self.streams:
            streams_list.add(stream._to_java())
        obj.streams = streams_list

        exchanges_map = LinkedHashMap()
        for code, overrides in self.exchanges.items():
            exchanges_map.put(str(code), overrides._to_java())
        obj.exchanges = exchanges_map
        if self.simulation_as_of is not None:
            obj.simulationAsOf = str(self.simulation_as_of)

        if self.strategy is not None:
            obj.strategy = self.strategy._to_java()
        obj.strategyId = jpype.JInt(self.strategy_id)
        obj.risk = self.risk._to_java()
        obj.record = self.record
        obj.recordDepth = jpype.JInt(self.record_depth)
        obj.measureProcessingTime = self.measure_processing_time
        if self.seed is not None:
            obj.seed = jpype.JLong(self.seed)

        return obj
