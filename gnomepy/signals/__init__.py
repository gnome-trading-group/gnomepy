from gnomepy.signals.registrable import Registrable
from gnomepy.signals.fill_rate_signal import FillRateSignal

from gnomepy.signals.volatility import VolatilitySignal, AdaptiveKalmanVolatility, SpreadVolatility
from gnomepy.signals.fair_value import FairValueModel, MidFairValue, MicropriceFairValue, DampenedFairValue, WeightedMicropriceFairValue

__all__ = [
    "Registrable",
    "FillRateSignal",
    "VolatilitySignal",
    "AdaptiveKalmanVolatility",
    "SpreadVolatility",
    "FairValueModel",
    "MidFairValue",
    "MicropriceFairValue",
    "DampenedFairValue",
    "WeightedMicropriceFairValue",
]


def get_registry(*args, **kwargs):
    """Lazy import of SignalRegistry to avoid requiring boto3 at import time."""
    from gnomepy.signals.registry import SignalRegistry
    return SignalRegistry(*args, **kwargs)
