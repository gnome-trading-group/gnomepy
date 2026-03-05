"""
Volatility-specific feature extraction for the LGBM volatility model.

30 features designed for predicting forward realized volatility, grouped into:
- Realized vol (8): rolling std and variance at multiple lookbacks
- Range estimators (3): Parkinson high-low range
- Vol dynamics (3): vol ratios and vol-of-vol
- Return structure (5): autocorrelation, abs return sums, kurtosis
- Microstructure (11): spread, depth, microprice, withdrawals

Provides two extraction modes maintaining train/serve parity:
- extract_vol_features_bulk: vectorized pandas for training
- extract_vol_features_tick: numpy for runtime inference
"""

import numpy as np
import pandas as pd

from gnomepy.research.data.mbp10.features import (
    compute_microstructure_bulk,
    compute_microstructure_tick,
    compute_returns_bulk,
    compute_returns_tick,
    compute_trade_flow_bulk,
    compute_trade_flow_tick,
)


# Minimum ticks needed before features can be computed.
# 50-tick vol window + 1 tick for diff + 1 for safety = 52
MIN_LOOKBACK = 52

VOL_FEATURE_NAMES: list[str] = [
    # Realized Vol (8)
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "volatility_50",
    "realized_var_5",
    "realized_var_10",
    "realized_var_20",
    "realized_var_50",
    # Range Estimators (3)
    "parkinson_10",
    "parkinson_20",
    "parkinson_50",
    # Vol Dynamics (3)
    "vol_ratio_5_20",
    "vol_ratio_5_50",
    "vol_of_vol_20",
    # Return Structure (5)
    "return_autocorr_10",
    "return_autocorr_20",
    "abs_return_sum_10",
    "abs_return_sum_20",
    "return_kurtosis_20",
    # Microstructure (11)
    "spread_bps",
    "spread_mean_20",
    "spread_std_10",
    "tob_imbalance",
    "depth_ratio",
    "total_bid_depth",
    "total_ask_depth",
    "depth_change_5",
    "microprice_deviation",
    "bid_withdrawal_5",
    "ask_withdrawal_5",
]

assert len(VOL_FEATURE_NAMES) == 30


# ---------------------------------------------------------------------------
# Bulk extraction (vectorized, for training)
# ---------------------------------------------------------------------------

def extract_vol_features_bulk(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 30 volatility features in a vectorized fashion over a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: bidPrice0..bidPrice9, askPrice0..askPrice9,
        bidSize0..bidSize9, askSize0..askSize9. Also needs 'midPrice' column
        or it will be computed from bidPrice0/askPrice0.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns matching VOL_FEATURE_NAMES. Rows where features
        cannot be computed (insufficient lookback) will contain NaN.
    """
    combined = pd.concat([
        compute_returns_bulk(df),
        compute_microstructure_bulk(df),
        compute_trade_flow_bulk(df),
    ], axis=1)
    return combined[VOL_FEATURE_NAMES]


# ---------------------------------------------------------------------------
# Single-tick extraction (for runtime inference)
# ---------------------------------------------------------------------------

def extract_vol_features_tick(listing_data: dict[str, np.ndarray]) -> np.ndarray | None:
    """Extract 30 volatility features for the most recent tick from OMS arrays.

    Parameters
    ----------
    listing_data : dict[str, np.ndarray]
        Dict mapping column names to numpy arrays of historical values.
        Arrays must have at least MIN_LOOKBACK elements.

    Returns
    -------
    np.ndarray or None
        1-D array of shape ``(30,)`` in the same order as
        ``VOL_FEATURE_NAMES``, or ``None`` if insufficient data.
    """
    bid0 = listing_data.get("bidPrice0")
    ask0 = listing_data.get("askPrice0")

    if bid0 is None or ask0 is None:
        return None
    if len(bid0) < MIN_LOOKBACK or len(ask0) < MIN_LOOKBACK:
        return None

    feats: dict[str, float] = {}
    feats.update(compute_returns_tick(listing_data))
    feats.update(compute_microstructure_tick(listing_data))
    feats.update(compute_trade_flow_tick(listing_data))

    return np.array([feats[name] for name in VOL_FEATURE_NAMES])
