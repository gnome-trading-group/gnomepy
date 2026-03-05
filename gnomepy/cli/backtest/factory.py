"""Build the full Backtest object graph from a validated config."""

import datetime

from gnomepy.cli.backtest.config import (
    BacktestConfig,
    BpsSignalConfig,
    GaussianLatencyConfig,
    KalmanBpsConfig,
    LGBMVolConfig,
    MarketMakingOMSConfig,
    MarketMakingSignalConfig,
    OptimisticQueueConfig,
    ProbabilisticQueueConfig,
    RealizedVolConfig,
    SimpleOMSConfig,
    StaticLatencyConfig,
)
from gnomepy.backtest.driver import Backtest
from gnomepy.backtest.exchanges.base import SimulatedExchange
from gnomepy.backtest.exchanges.mbp.mbp import MBPSimulatedExchange
from gnomepy.backtest.fee import StaticFeeModel
from gnomepy.backtest.latency import GaussianLatency, LatencyModel, StaticLatency
from gnomepy.backtest.queues import (
    OptimisticQueueModel,
    ProbabilisticQueueModel,
    RiskAverseQueueModel,
)
from gnomepy.data.types import SchemaType
from gnomepy.registry.api import RegistryClient
from gnomepy.research.models.bps.kalman.model import KalmanBpsModel
from gnomepy.research.models.volatility.lgbm.model import LGBMVolatilityModel
from gnomepy.research.models.volatility.realized.model import RealizedVolatilityModel
from gnomepy.research.oms.market_making import MarketMakingOMS
from gnomepy.research.oms.simple import SimpleOMS
from gnomepy.research.signals.bps.signal import BpsSignal
from gnomepy.research.signals.market_making.signal import MarketMakingSignal
from gnomepy.research.strategies.simple import SimpleStrategy


def build_backtest(config: BacktestConfig) -> Backtest:
    """Construct a fully wired ``Backtest`` from a validated config."""
    bt = config.backtest
    start = datetime.datetime.fromisoformat(bt.start_datetime)
    end = datetime.datetime.fromisoformat(bt.end_datetime)
    listing_ids = bt.listing_ids
    schema_type = SchemaType(bt.schema_type)

    registry_client = RegistryClient()

    # Resolve listings so signals can reference Listing objects
    listings_by_id = {}
    for lid in listing_ids:
        result = registry_client.get_listing(listing_id=lid)
        if not result:
            raise ValueError(f"Listing ID {lid} not found in registry")
        listings_by_id[lid] = result[0]

    # -- Exchange --
    ex = config.exchange.default
    exchange = MBPSimulatedExchange(
        fee_model=StaticFeeModel(taker_fee=ex.fee_model.taker_fee, maker_fee=ex.fee_model.maker_fee),
        network_latency=_build_latency(ex.network_latency),
        order_processing_latency=_build_latency(ex.order_processing_latency),
        queue_model=_build_queue_model(ex.queue_model),
    )

    exchanges: dict[int, dict[int, SimulatedExchange]] = {}
    for listing in listings_by_id.values():
        exchanges.setdefault(listing.exchange_id, {})[listing.security_id] = exchange

    # -- Signals --
    signals = [_build_signal(sig, listings_by_id) for sig in config.signals]

    # -- OMS --
    oms = _build_oms(config.oms, signals)

    # -- Strategy --
    processing_latency = _build_latency(config.strategy.processing_latency)
    strategy = SimpleStrategy(processing_latency=processing_latency, oms=oms)

    return Backtest(
        start_datetime=start,
        end_datetime=end,
        listing_ids=listing_ids,
        schema_type=schema_type,
        strategy=strategy,
        exchanges=exchanges,
        registry_client=registry_client,
    )


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------

def _build_latency(cfg: StaticLatencyConfig | GaussianLatencyConfig) -> LatencyModel:
    match cfg:
        case StaticLatencyConfig():
            return StaticLatency(latency=cfg.latency_ns)
        case GaussianLatencyConfig():
            return GaussianLatency(mu=cfg.mu, sigma=cfg.sigma)


def _build_queue_model(cfg):
    match cfg:
        case OptimisticQueueConfig():
            return OptimisticQueueModel()
        case ProbabilisticQueueConfig():
            return ProbabilisticQueueModel(cancel_ahead_probability=cfg.cancel_ahead_probability)
        case _:
            return RiskAverseQueueModel()


def _build_signal(cfg: MarketMakingSignalConfig | BpsSignalConfig, listings_by_id: dict):
    listing = listings_by_id[cfg.listing_id]

    match cfg:
        case MarketMakingSignalConfig():
            vol_cfg = cfg.volatility_model
            match vol_cfg:
                case RealizedVolConfig():
                    vol_model = RealizedVolatilityModel(window=vol_cfg.window, horizon=vol_cfg.horizon)
                case LGBMVolConfig():
                    vol_model = LGBMVolatilityModel(model_path=vol_cfg.model_path, horizon=vol_cfg.horizon)

            return MarketMakingSignal(
                listing=listing,
                volatility_model=vol_model,
                vol_threshold_bps=cfg.vol_threshold_bps,
                trade_frequency=cfg.trade_frequency,
                gamma=cfg.gamma,
                order_arrival_rate=cfg.order_arrival_rate,
                volatility_window=cfg.volatility_window,
                volatility_half_life=cfg.volatility_half_life,
                max_inventory=cfg.max_inventory,
                liquidation_threshold=cfg.liquidation_threshold,
                use_market_orders_for_liquidation=cfg.use_market_orders_for_liquidation,
                min_spread_bps=cfg.min_spread_bps,
                max_spread_ticks=cfg.max_spread_ticks,
                min_volatility=cfg.min_volatility,
            )

        case BpsSignalConfig():
            bps_model = None
            if cfg.model is not None:
                match cfg.model:
                    case KalmanBpsConfig():
                        bps_model = KalmanBpsModel(
                            window=cfg.model.window,
                            horizon=cfg.model.horizon,
                            sample_interval=cfg.model.sample_interval,
                            observation_noise=cfg.model.observation_noise,
                            process_noise_level=cfg.model.process_noise_level,
                        )

            return BpsSignal(
                listing=listing,
                model=bps_model,
                trade_frequency=cfg.trade_frequency,
                entry_threshold_bps=cfg.entry_threshold_bps,
                transaction_cost_bps=cfg.transaction_cost_bps,
                max_inventory=cfg.max_inventory,
                exit_hold_ticks=cfg.exit_hold_ticks,
                stop_loss_bps=cfg.stop_loss_bps,
                take_profit_bps=cfg.take_profit_bps,
            )


def _build_oms(cfg: SimpleOMSConfig | MarketMakingOMSConfig, signals: list):
    match cfg:
        case SimpleOMSConfig():
            return SimpleOMS(signals=signals, notional=cfg.notional, starting_cash=cfg.starting_cash)
        case MarketMakingOMSConfig():
            return MarketMakingOMS(
                signals=signals,
                notional=cfg.notional,
                starting_cash=cfg.starting_cash,
                max_position_notional=cfg.max_position_notional,
                position_aware_sizing=cfg.position_aware_sizing,
                position_scaling_factor=cfg.position_scaling_factor,
                tick_size=cfg.tick_size,
                passive_reprice_ticks=cfg.passive_reprice_ticks,
            )
