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


def _safe_imbalance(bid_val: float, ask_val: float) -> float:
    total = bid_val + ask_val
    if total == 0:
        return 0.0
    return (bid_val - ask_val) / total


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
    out = pd.DataFrame(index=df.index)

    if "midPrice" not in df.columns:
        df = df.copy()
        df["midPrice"] = (df["bidPrice0"] + df["askPrice0"]) / 2.0

    mid = df["midPrice"]
    log_mid = np.log(mid)
    log_returns = log_mid.diff(1)

    # ------------------------------------------------------------------
    # Realized Vol (8)
    # ------------------------------------------------------------------
    for window in [5, 10, 20, 50]:
        out[f"volatility_{window}"] = log_returns.rolling(window).std()
        out[f"realized_var_{window}"] = (log_returns ** 2).rolling(window).sum()

    # ------------------------------------------------------------------
    # Range Estimators — Parkinson (3)
    # ------------------------------------------------------------------
    # Parkinson = sqrt(1/(4*n*ln2) * sum(ln(H/L)^2))
    # We normalize by using rolling mid as denominator for high-low range
    high = df[["bidPrice0", "askPrice0"]].max(axis=1)
    low = df[["bidPrice0", "askPrice0"]].min(axis=1)
    log_hl_sq = (np.log(high / low.replace(0, np.nan))) ** 2

    for window in [10, 20, 50]:
        out[f"parkinson_{window}"] = np.sqrt(
            log_hl_sq.rolling(window).mean() / (4 * np.log(2))
        )

    # ------------------------------------------------------------------
    # Vol Dynamics (3)
    # ------------------------------------------------------------------
    vol_5 = out["volatility_5"]
    vol_20 = out["volatility_20"]
    vol_50 = out["volatility_50"]

    out["vol_ratio_5_20"] = (vol_5 / vol_20.replace(0, np.nan)).fillna(1.0)
    out["vol_ratio_5_50"] = (vol_5 / vol_50.replace(0, np.nan)).fillna(1.0)
    out["vol_of_vol_20"] = vol_5.rolling(20).std()

    # ------------------------------------------------------------------
    # Return Structure (5)
    # ------------------------------------------------------------------
    for window in [10, 20]:
        out[f"return_autocorr_{window}"] = log_returns.rolling(window).apply(
            lambda x: pd.Series(x).autocorr(lag=1), raw=False
        ).fillna(0.0)

    for window in [10, 20]:
        out[f"abs_return_sum_{window}"] = log_returns.abs().rolling(window).sum()

    # Excess kurtosis (bias-corrected, matching pandas)
    out["return_kurtosis_20"] = log_returns.rolling(20).apply(
        lambda x: pd.Series(x).kurtosis(), raw=False
    ).fillna(0.0)

    # ------------------------------------------------------------------
    # Microstructure (11)
    # ------------------------------------------------------------------
    spread = df["askPrice0"] - df["bidPrice0"]
    spread_bps = (spread / mid) * 1e4
    out["spread_bps"] = spread_bps
    out["spread_mean_20"] = spread_bps.rolling(20).mean()
    out["spread_std_10"] = spread_bps.rolling(10).std()

    # TOB imbalance
    bid_sz0 = df["bidSize0"]
    ask_sz0 = df["askSize0"]
    out["tob_imbalance"] = (bid_sz0 - ask_sz0) / (bid_sz0 + ask_sz0).replace(0, np.nan)
    out["tob_imbalance"] = out["tob_imbalance"].fillna(0.0)

    # Depth features
    total_bid: pd.Series = sum(df[f"bidSize{i}"] for i in range(10))  # type: ignore[assignment]
    total_ask: pd.Series = sum(df[f"askSize{i}"] for i in range(10))  # type: ignore[assignment]
    depth_total = total_bid + total_ask
    out["depth_ratio"] = (total_bid / depth_total.replace(0, np.nan)).fillna(0.5)
    out["total_bid_depth"] = total_bid
    out["total_ask_depth"] = total_ask
    out["depth_change_5"] = (total_bid - total_ask).diff(5)

    # Microprice deviation
    sz_total = bid_sz0 + ask_sz0
    microprice = (df["bidPrice0"] * ask_sz0 + df["askPrice0"] * bid_sz0) / sz_total.replace(0, np.nan)
    microprice = microprice.fillna(mid)
    out["microprice_deviation"] = (microprice - mid) / mid

    # Withdrawal features
    for side, prefix in [("bid", "bid"), ("ask", "ask")]:
        total_withdrawal: pd.Series = sum(  # type: ignore[assignment]
            df[f"{prefix}Size{i}"].diff(1).clip(upper=0) for i in range(10)
        ).abs()  # type: ignore[union-attr]
        out[f"{side}_withdrawal_5"] = total_withdrawal.rolling(5).sum()

    # Reorder columns to match VOL_FEATURE_NAMES exactly
    return out[VOL_FEATURE_NAMES]


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

    features = np.empty(len(VOL_FEATURE_NAMES), dtype=np.float64)

    n = len(bid0)
    mid_history = (bid0 + ask0) / 2.0
    log_mid = np.log(mid_history)
    log_returns = np.diff(log_mid)  # length n-1

    def _last(key: str) -> float:
        arr = listing_data.get(key)
        if arr is not None and len(arr) > 0:
            return float(arr[-1])
        return 0.0

    # ------------------------------------------------------------------
    # Realized Vol (8): indices 0-7
    # ------------------------------------------------------------------
    for i, window in enumerate([5, 10, 20, 50]):
        if len(log_returns) >= window:
            features[i] = float(np.std(log_returns[-window:], ddof=1))
            features[4 + i] = float(np.sum(log_returns[-window:] ** 2))
        else:
            features[i] = np.nan
            features[4 + i] = np.nan

    # ------------------------------------------------------------------
    # Range Estimators — Parkinson (3): indices 8-10
    # ------------------------------------------------------------------
    high = np.maximum(bid0, ask0)
    low = np.minimum(bid0, ask0)
    # Avoid log(0)
    safe_low = np.where(low == 0, 1, low)
    log_hl_sq = np.log(high / safe_low) ** 2

    for i, window in enumerate([10, 20, 50]):
        if n >= window:
            features[8 + i] = float(np.sqrt(
                np.mean(log_hl_sq[-window:]) / (4 * np.log(2))
            ))
        else:
            features[8 + i] = np.nan

    # ------------------------------------------------------------------
    # Vol Dynamics (3): indices 11-13
    # ------------------------------------------------------------------
    vol_5 = features[0]   # volatility_5
    vol_20 = features[2]  # volatility_20
    vol_50 = features[3]  # volatility_50

    # vol_ratio_5_20
    features[11] = (vol_5 / vol_20) if vol_20 != 0 and not np.isnan(vol_20) else 1.0
    # vol_ratio_5_50
    features[12] = (vol_5 / vol_50) if vol_50 != 0 and not np.isnan(vol_50) else 1.0

    # vol_of_vol_20: std of rolling volatility_5 over last 20 values
    # We need at least 20 windows of vol_5, so 5 + 20 - 1 = 24 returns
    if len(log_returns) >= 24:
        # Compute vol_5 at each of last 20 positions
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
            features[13] = float(np.std(vol_5_arr[valid], ddof=1))
        else:
            features[13] = np.nan
    else:
        features[13] = np.nan

    # ------------------------------------------------------------------
    # Return Structure (5): indices 14-18
    # ------------------------------------------------------------------
    # return_autocorr_10, return_autocorr_20
    for i, window in enumerate([10, 20]):
        if len(log_returns) >= window:
            x = log_returns[-window:]
            # autocorrelation at lag 1 using np.corrcoef
            if len(x) > 1:
                x1 = x[:-1]
                x2 = x[1:]
                if np.std(x1) > 0 and np.std(x2) > 0:
                    features[14 + i] = float(np.corrcoef(x1, x2)[0, 1])
                else:
                    features[14 + i] = 0.0
            else:
                features[14 + i] = 0.0
        else:
            features[14 + i] = 0.0

    # abs_return_sum_10, abs_return_sum_20
    for i, window in enumerate([10, 20]):
        if len(log_returns) >= window:
            features[16 + i] = float(np.sum(np.abs(log_returns[-window:])))
        else:
            features[16 + i] = np.nan

    # return_kurtosis_20 (bias-corrected excess kurtosis matching pandas)
    if len(log_returns) >= 20:
        x = log_returns[-20:]
        nn = len(x)
        mean = np.mean(x)
        std = np.std(x, ddof=1)
        if std > 0 and nn > 3:
            # Bias-corrected excess kurtosis (Fisher's definition, matching pandas)
            kurt = ((nn * (nn + 1)) / ((nn - 1) * (nn - 2) * (nn - 3)) *
                    np.sum(((x - mean) / std) ** 4) -
                    3 * (nn - 1) ** 2 / ((nn - 2) * (nn - 3)))
            features[18] = float(kurt)
        else:
            features[18] = 0.0
    else:
        features[18] = 0.0

    # ------------------------------------------------------------------
    # Microstructure (11): indices 19-29
    # ------------------------------------------------------------------
    cur_bid = float(bid0[-1])
    cur_ask = float(ask0[-1])
    cur_mid = (cur_bid + cur_ask) / 2.0

    # spread_bps
    spread_val = cur_ask - cur_bid
    features[19] = (spread_val / cur_mid) * 1e4 if cur_mid != 0 else 0.0

    # spread_mean_20
    if n >= 20:
        spread_hist = (ask0[-20:] - bid0[-20:])
        mid_hist = (bid0[-20:] + ask0[-20:]) / 2.0
        spread_bps_hist = np.where(mid_hist == 0, 0, spread_hist / mid_hist * 1e4)
        features[20] = float(np.mean(spread_bps_hist))
    else:
        features[20] = features[19]

    # spread_std_10
    if n >= 10:
        spread_hist_10 = (ask0[-10:] - bid0[-10:])
        mid_hist_10 = (bid0[-10:] + ask0[-10:]) / 2.0
        spread_bps_10 = np.where(mid_hist_10 == 0, 0, spread_hist_10 / mid_hist_10 * 1e4)
        features[21] = float(np.std(spread_bps_10, ddof=1))
    else:
        features[21] = np.nan

    # tob_imbalance
    bs0 = _last("bidSize0")
    as0 = _last("askSize0")
    features[22] = _safe_imbalance(bs0, as0)

    # depth_ratio, total_bid_depth, total_ask_depth
    total_bid = sum(_last(f"bidSize{i}") for i in range(10))
    total_ask = sum(_last(f"askSize{i}") for i in range(10))
    depth_total = total_bid + total_ask
    features[23] = (total_bid / depth_total) if depth_total != 0 else 0.5
    features[24] = total_bid
    features[25] = total_ask

    # depth_change_5
    if n >= 6:
        prev_depth_diff = 0.0
        for i in range(10):
            b_arr = listing_data.get(f"bidSize{i}")
            a_arr = listing_data.get(f"askSize{i}")
            if b_arr is not None and len(b_arr) >= 6:
                prev_depth_diff += float(b_arr[-6])
            if a_arr is not None and len(a_arr) >= 6:
                prev_depth_diff -= float(a_arr[-6])
        features[26] = (total_bid - total_ask) - prev_depth_diff
    else:
        features[26] = np.nan

    # microprice_deviation
    sz_total = bs0 + as0
    if sz_total > 0:
        microprice = (cur_bid * as0 + cur_ask * bs0) / sz_total
    else:
        microprice = cur_mid
    features[27] = (microprice - cur_mid) / cur_mid if cur_mid != 0 else 0.0

    # bid_withdrawal_5, ask_withdrawal_5
    if n >= 6:
        bid_withdrawal = np.zeros(n - 1)
        ask_withdrawal = np.zeros(n - 1)
        for i in range(10):
            b_arr = listing_data.get(f"bidSize{i}")
            if b_arr is not None and len(b_arr) >= n:
                bid_withdrawal += np.abs(np.minimum(np.diff(b_arr[-n:]), 0))
            a_arr = listing_data.get(f"askSize{i}")
            if a_arr is not None and len(a_arr) >= n:
                ask_withdrawal += np.abs(np.minimum(np.diff(a_arr[-n:]), 0))

        features[28] = float(bid_withdrawal[-5:].sum()) if len(bid_withdrawal) >= 5 else float(bid_withdrawal.sum())
        features[29] = float(ask_withdrawal[-5:].sum()) if len(ask_withdrawal) >= 5 else float(ask_withdrawal.sum())
    else:
        features[28] = np.nan
        features[29] = np.nan

    return features
