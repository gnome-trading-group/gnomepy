"""
Returns and volatility features for MBP10 numpy-array data.

Covers time-series features derived from mid-price log returns:
- Log returns at multiple lookbacks (5, 10, 20, 50)
- Rolling volatility (std of log returns)
- Realized variance (sum of squared returns)
- Parkinson range estimator
- Vol ratios and vol-of-vol
- Return autocorrelation
- Absolute return sums
- Return kurtosis

Provides two extraction modes:
- compute_returns_bulk: vectorized over a DataFrame (for training)
- compute_returns_tick: single-observation from numpy arrays (for inference)
"""

import numpy as np
import pandas as pd


def compute_returns_bulk(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all returns/volatility features over a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: bidPrice0, askPrice0. Also needs 'midPrice'
        column or it will be computed from bidPrice0/askPrice0.

    Returns
    -------
    pd.DataFrame
        DataFrame with all returns feature columns. Rows where features
        cannot be computed (insufficient lookback) will contain NaN.
    """
    out = pd.DataFrame(index=df.index)

    if "midPrice" not in df.columns:
        df = df.copy()
        df["midPrice"] = (df["bidPrice0"] + df["askPrice0"]) / 2.0

    mid = df["midPrice"]
    log_mid = np.log(mid)
    log_returns = log_mid.diff(1)

    # ------------------------------------------------------------------
    # Log returns at multiple lookbacks
    # ------------------------------------------------------------------
    for window in [5, 10, 20, 50]:
        out[f"return_{window}"] = log_mid.diff(window)

    # ------------------------------------------------------------------
    # Realized Vol and Variance
    # ------------------------------------------------------------------
    for window in [5, 10, 20, 50]:
        out[f"volatility_{window}"] = log_returns.rolling(window).std()
        out[f"realized_var_{window}"] = (log_returns ** 2).rolling(window).sum()

    # ------------------------------------------------------------------
    # Range Estimators — Parkinson
    # ------------------------------------------------------------------
    high = df[["bidPrice0", "askPrice0"]].max(axis=1)
    low = df[["bidPrice0", "askPrice0"]].min(axis=1)
    log_hl_sq = (np.log(high / low.replace(0, np.nan))) ** 2

    for window in [10, 20, 50]:
        out[f"parkinson_{window}"] = np.sqrt(
            log_hl_sq.rolling(window).mean() / (4 * np.log(2))
        )

    # ------------------------------------------------------------------
    # Vol Dynamics
    # ------------------------------------------------------------------
    vol_5 = out["volatility_5"]
    vol_20 = out["volatility_20"]
    vol_50 = out["volatility_50"]

    out["vol_ratio_5_20"] = (vol_5 / vol_20.replace(0, np.nan)).fillna(1.0)
    out["vol_ratio_5_50"] = (vol_5 / vol_50.replace(0, np.nan)).fillna(1.0)
    out["vol_of_vol_20"] = vol_5.rolling(20).std()

    # ------------------------------------------------------------------
    # Return Structure
    # ------------------------------------------------------------------
    for window in [10, 20]:
        out[f"return_autocorr_{window}"] = log_returns.rolling(window).apply(
            lambda x: pd.Series(x).autocorr(lag=1), raw=False
        ).fillna(0.0)

    for window in [10, 20]:
        out[f"abs_return_sum_{window}"] = log_returns.abs().rolling(window).sum()

    out["return_kurtosis_20"] = log_returns.rolling(20).apply(
        lambda x: pd.Series(x).kurtosis(), raw=False
    ).fillna(0.0)

    return out


def compute_returns_tick(listing_data: dict[str, np.ndarray]) -> dict[str, float]:
    """Extract all returns/volatility features for the most recent tick.

    Parameters
    ----------
    listing_data : dict[str, np.ndarray]
        Dict mapping column names to numpy arrays of historical values.

    Returns
    -------
    dict[str, float]
        Feature name -> value mapping for all returns features.
    """
    bid0 = listing_data.get("bidPrice0")
    ask0 = listing_data.get("askPrice0")

    out: dict[str, float] = {}

    if bid0 is None or ask0 is None:
        # Return NaN for all features
        for window in [5, 10, 20, 50]:
            out[f"return_{window}"] = np.nan
            out[f"volatility_{window}"] = np.nan
            out[f"realized_var_{window}"] = np.nan
        for window in [10, 20, 50]:
            out[f"parkinson_{window}"] = np.nan
        out["vol_ratio_5_20"] = np.nan
        out["vol_ratio_5_50"] = np.nan
        out["vol_of_vol_20"] = np.nan
        for window in [10, 20]:
            out[f"return_autocorr_{window}"] = np.nan
            out[f"abs_return_sum_{window}"] = np.nan
        out["return_kurtosis_20"] = np.nan
        return out

    n = len(bid0)
    mid_history = (bid0 + ask0) / 2.0
    log_mid = np.log(mid_history)
    log_returns = np.diff(log_mid)  # length n-1

    # --- Log returns ---
    for window in [5, 10, 20, 50]:
        if n >= window + 1:
            out[f"return_{window}"] = log_mid[-1] - log_mid[-(window + 1)]
        else:
            out[f"return_{window}"] = np.nan

    # --- Realized vol and variance ---
    for window in [5, 10, 20, 50]:
        if len(log_returns) >= window:
            out[f"volatility_{window}"] = float(np.std(log_returns[-window:], ddof=1))
            out[f"realized_var_{window}"] = float(np.sum(log_returns[-window:] ** 2))
        else:
            out[f"volatility_{window}"] = np.nan
            out[f"realized_var_{window}"] = np.nan

    # --- Parkinson ---
    high = np.maximum(bid0, ask0)
    low = np.minimum(bid0, ask0)
    safe_low = np.where(low == 0, 1, low)
    log_hl_sq = np.log(high / safe_low) ** 2

    for window in [10, 20, 50]:
        if n >= window:
            out[f"parkinson_{window}"] = float(np.sqrt(
                np.mean(log_hl_sq[-window:]) / (4 * np.log(2))
            ))
        else:
            out[f"parkinson_{window}"] = np.nan

    # --- Vol dynamics ---
    vol_5 = out["volatility_5"]
    vol_20 = out["volatility_20"]
    vol_50 = out["volatility_50"]

    out["vol_ratio_5_20"] = (vol_5 / vol_20) if (not np.isnan(vol_20) and vol_20 != 0) else 1.0
    out["vol_ratio_5_50"] = (vol_5 / vol_50) if (not np.isnan(vol_50) and vol_50 != 0) else 1.0

    if len(log_returns) >= 24:
        vol_5_arr = np.empty(20)
        for j in range(20):
            end = len(log_returns) - (19 - j)
            start = end - 5
            if start >= 0:
                vol_5_arr[j] = float(np.std(log_returns[start:end], ddof=1))
            else:
                vol_5_arr[j] = np.nan
        valid = ~np.isnan(vol_5_arr)
        if valid.sum() >= 2:
            out["vol_of_vol_20"] = float(np.std(vol_5_arr[valid], ddof=1))
        else:
            out["vol_of_vol_20"] = np.nan
    else:
        out["vol_of_vol_20"] = np.nan

    # --- Return autocorrelation ---
    for window in [10, 20]:
        if len(log_returns) >= window:
            x = log_returns[-window:]
            if len(x) > 1:
                x1 = x[:-1]
                x2 = x[1:]
                if np.std(x1) > 0 and np.std(x2) > 0:
                    out[f"return_autocorr_{window}"] = float(np.corrcoef(x1, x2)[0, 1])
                else:
                    out[f"return_autocorr_{window}"] = 0.0
            else:
                out[f"return_autocorr_{window}"] = 0.0
        else:
            out[f"return_autocorr_{window}"] = 0.0

    # --- Abs return sums ---
    for window in [10, 20]:
        if len(log_returns) >= window:
            out[f"abs_return_sum_{window}"] = float(np.sum(np.abs(log_returns[-window:])))
        else:
            out[f"abs_return_sum_{window}"] = np.nan

    # --- Return kurtosis ---
    if len(log_returns) >= 20:
        x = log_returns[-20:]
        nn = len(x)
        mean = np.mean(x)
        std = np.std(x, ddof=1)
        if std > 0 and nn > 3:
            kurt = ((nn * (nn + 1)) / ((nn - 1) * (nn - 2) * (nn - 3)) *
                    np.sum(((x - mean) / std) ** 4) -
                    3 * (nn - 1) ** 2 / ((nn - 2) * (nn - 3)))
            out["return_kurtosis_20"] = float(kurt)
        else:
            out["return_kurtosis_20"] = 0.0
    else:
        out["return_kurtosis_20"] = 0.0

    return out
