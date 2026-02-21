"""
Feature extraction for LightGBM directional prediction.

Operates directly on OMS numpy arrays (dict[str, np.ndarray] with keys like
bidPrice0..bidPrice9, askSize0..askSize9, etc). Does NOT reuse
data/features/mbp10/imbalance.py since those operate on MBP10 dataclass objects.

Provides two extraction modes:
- extract_features_bulk: vectorized over a DataFrame for training
- extract_features_tick: single-tick extraction for runtime inference
"""

import numpy as np
import pandas as pd


# Minimum ticks needed before features can be computed
# 200-tick z-score window + 1
MIN_LOOKBACK = 201

# Canonical feature names -- ensures train/serve parity
FEATURE_NAMES: list[str] = [
    # Imbalance (6)
    "tob_imbalance",
    "depth3_imbalance",
    "depth5_imbalance",
    "depth10_imbalance",
    "count_imbalance",
    "price_weighted_imbalance",
    # Depth (4)
    "total_bid_depth",
    "total_ask_depth",
    "depth_ratio",
    "depth_diff",
    # Spread / Microprice (4)
    "spread",
    "spread_bps",
    "microprice",
    "microprice_deviation",
    # Momentum (8)
    "return_5",
    "return_10",
    "return_20",
    "return_50",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "volatility_50",
    # Dynamics (4)
    "imbalance_change_5",
    "spread_change_5",
    "depth_change_5",
    "vwap_deviation",
    # Trade flow (6)
    "inferred_buy_vol_5",
    "inferred_buy_vol_20",
    "inferred_sell_vol_5",
    "inferred_sell_vol_20",
    "trade_imbalance_5",
    "trade_imbalance_20",
    # Z-score normalized (6)
    "tob_imbalance_z",
    "depth_ratio_z",
    "total_bid_depth_z",
    "total_ask_depth_z",
    "spread_bps_z",
    "depth_diff_z",
    # Depth change decomposition (6)
    "bid_withdrawal_5",
    "ask_withdrawal_5",
    "withdrawal_imbalance_5",
    "bid_withdrawal_20",
    "ask_withdrawal_20",
    "withdrawal_imbalance_20",
]


def _safe_imbalance(bid_val: float, ask_val: float) -> float:
    total = bid_val + ask_val
    if total == 0:
        return 0.0
    return (bid_val - ask_val) / total


# ---------------------------------------------------------------------------
# Bulk extraction (vectorized, for training)
# ---------------------------------------------------------------------------

def extract_features_bulk(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features in a vectorized fashion over a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: bidPrice0..bidPrice9, askPrice0..askPrice9,
        bidSize0..bidSize9, askSize0..askSize9, bidCount0..bidCount9,
        askCount0..askCount9 (all as floats). Also needs 'midPrice' column
        or it will be computed from bidPrice0/askPrice0.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns matching FEATURE_NAMES. Rows where features
        cannot be computed (e.g. insufficient lookback) will contain NaN.
    """
    out = pd.DataFrame(index=df.index)

    # Ensure midPrice exists
    if "midPrice" not in df.columns:
        df = df.copy()
        df["midPrice"] = (df["bidPrice0"] + df["askPrice0"]) / 2.0

    mid = df["midPrice"]

    # ------------------------------------------------------------------
    # Imbalance features
    # ------------------------------------------------------------------
    # TOB imbalance
    bid_sz0 = df["bidSize0"]
    ask_sz0 = df["askSize0"]
    out["tob_imbalance"] = (bid_sz0 - ask_sz0) / (bid_sz0 + ask_sz0).replace(0, np.nan)
    out["tob_imbalance"] = out["tob_imbalance"].fillna(0.0)

    # Depth-N imbalance helper
    def _depth_imbalance(n: int) -> pd.Series:
        bid_sum = sum(df[f"bidSize{i}"] for i in range(n))
        ask_sum = sum(df[f"askSize{i}"] for i in range(n))
        total = bid_sum + ask_sum
        return ((bid_sum - ask_sum) / total.replace(0, np.nan)).fillna(0.0)

    out["depth3_imbalance"] = _depth_imbalance(3)
    out["depth5_imbalance"] = _depth_imbalance(5)
    out["depth10_imbalance"] = _depth_imbalance(10)

    # Count imbalance (TOB)
    bid_ct0 = df.get("bidCount0", pd.Series(0, index=df.index))
    ask_ct0 = df.get("askCount0", pd.Series(0, index=df.index))
    ct_total = bid_ct0 + ask_ct0
    out["count_imbalance"] = ((bid_ct0 - ask_ct0) / ct_total.replace(0, np.nan)).fillna(0.0)

    # Price-weighted imbalance (all 10 levels)
    pw_bid = sum(df[f"bidPrice{i}"] * df[f"bidSize{i}"] for i in range(10))
    pw_ask = sum(df[f"askPrice{i}"] * df[f"askSize{i}"] for i in range(10))
    pw_total = pw_bid + pw_ask
    out["price_weighted_imbalance"] = ((pw_bid - pw_ask) / pw_total.replace(0, np.nan)).fillna(0.0)

    # ------------------------------------------------------------------
    # Depth features
    # ------------------------------------------------------------------
    total_bid = sum(df[f"bidSize{i}"] for i in range(10))
    total_ask = sum(df[f"askSize{i}"] for i in range(10))
    out["total_bid_depth"] = total_bid
    out["total_ask_depth"] = total_ask
    depth_total = total_bid + total_ask
    out["depth_ratio"] = (total_bid / depth_total.replace(0, np.nan)).fillna(0.5)
    out["depth_diff"] = total_bid - total_ask

    # ------------------------------------------------------------------
    # Spread / Microprice features
    # ------------------------------------------------------------------
    spread = df["askPrice0"] - df["bidPrice0"]
    out["spread"] = spread
    out["spread_bps"] = (spread / mid) * 1e4

    # Size-weighted microprice
    microprice = (df["bidPrice0"] * ask_sz0 + df["askPrice0"] * bid_sz0) / (bid_sz0 + ask_sz0).replace(0, np.nan)
    microprice = microprice.fillna(mid)
    out["microprice"] = microprice
    out["microprice_deviation"] = (microprice - mid) / mid

    # ------------------------------------------------------------------
    # Momentum features (returns & volatility at multiple lookbacks)
    # ------------------------------------------------------------------
    log_mid = np.log(mid)

    for window in [5, 10, 20, 50]:
        out[f"return_{window}"] = log_mid.diff(window)
        out[f"volatility_{window}"] = log_mid.diff(1).rolling(window).std()

    # ------------------------------------------------------------------
    # Dynamics features
    # ------------------------------------------------------------------
    out["imbalance_change_5"] = out["tob_imbalance"].diff(5)
    out["spread_change_5"] = out["spread_bps"].diff(5)
    out["depth_change_5"] = (total_bid - total_ask).diff(5)

    # VWAP deviation (approximate VWAP from top 5 levels)
    vwap_num = sum(df[f"bidPrice{i}"] * df[f"bidSize{i}"] + df[f"askPrice{i}"] * df[f"askSize{i}"] for i in range(5))
    vwap_den = sum(df[f"bidSize{i}"] + df[f"askSize{i}"] for i in range(5))
    vwap = (vwap_num / vwap_den.replace(0, np.nan)).fillna(mid)
    out["vwap_deviation"] = (mid - vwap) / mid

    # ------------------------------------------------------------------
    # Trade flow (book-inferred)
    # ------------------------------------------------------------------
    same_ask = df["askPrice0"] == df["askPrice0"].shift(1)
    ask_decrease = (df["askSize0"].shift(1) - df["askSize0"]).clip(lower=0)
    inferred_buy = np.where(same_ask, ask_decrease, 0)

    same_bid = df["bidPrice0"] == df["bidPrice0"].shift(1)
    bid_decrease = (df["bidSize0"].shift(1) - df["bidSize0"]).clip(lower=0)
    inferred_sell = np.where(same_bid, bid_decrease, 0)

    buy_series = pd.Series(inferred_buy, index=df.index)
    sell_series = pd.Series(inferred_sell, index=df.index)
    for w in [5, 20]:
        out[f"inferred_buy_vol_{w}"] = buy_series.rolling(w).sum()
        out[f"inferred_sell_vol_{w}"] = sell_series.rolling(w).sum()
        total_flow = out[f"inferred_buy_vol_{w}"] + out[f"inferred_sell_vol_{w}"]
        out[f"trade_imbalance_{w}"] = (
            (out[f"inferred_buy_vol_{w}"] - out[f"inferred_sell_vol_{w}"])
            / total_flow.replace(0, np.nan)
        ).fillna(0)

    # ------------------------------------------------------------------
    # Z-score normalization (200-tick rolling window)
    # ------------------------------------------------------------------
    z_window = 200
    z_sources = {
        "tob_imbalance_z": "tob_imbalance",
        "depth_ratio_z": "depth_ratio",
        "total_bid_depth_z": "total_bid_depth",
        "total_ask_depth_z": "total_ask_depth",
        "spread_bps_z": "spread_bps",
        "depth_diff_z": "depth_diff",
    }
    for z_name, raw_name in z_sources.items():
        rolling_mean = out[raw_name].rolling(z_window).mean()
        rolling_std = out[raw_name].rolling(z_window).std().replace(0, 1)
        out[z_name] = (out[raw_name] - rolling_mean) / rolling_std

    # ------------------------------------------------------------------
    # Depth change decomposition
    # ------------------------------------------------------------------
    for side, prefix in [("bid", "bid"), ("ask", "ask")]:
        total_withdrawal = sum(
            df[f"{prefix}Size{i}"].diff(1).clip(upper=0) for i in range(10)
        ).abs()
        for w in [5, 20]:
            out[f"{side}_withdrawal_{w}"] = total_withdrawal.rolling(w).sum()

    for w in [5, 20]:
        wd_total = out[f"bid_withdrawal_{w}"] + out[f"ask_withdrawal_{w}"]
        out[f"withdrawal_imbalance_{w}"] = (
            (out[f"bid_withdrawal_{w}"] - out[f"ask_withdrawal_{w}"])
            / wd_total.replace(0, np.nan)
        ).fillna(0)

    # Reorder columns to match FEATURE_NAMES exactly
    return out[FEATURE_NAMES]


# ---------------------------------------------------------------------------
# Single-tick extraction (for runtime inference)
# ---------------------------------------------------------------------------

def extract_features_tick(listing_data: dict[str, np.ndarray]) -> np.ndarray | None:
    """Extract features for the most recent tick from OMS numpy arrays.

    Parameters
    ----------
    listing_data : dict[str, np.ndarray]
        Dict mapping column names to numpy arrays of historical values, e.g.
        ``{"bidPrice0": np.array([...]), "askPrice0": np.array([...]), ...}``.
        Arrays must have at least MIN_LOOKBACK elements.

    Returns
    -------
    np.ndarray or None
        1-D array of shape ``(len(FEATURE_NAMES),)`` in the same order as
        ``FEATURE_NAMES``, or ``None`` if insufficient data.
    """
    bid0 = listing_data.get("bidPrice0")
    ask0 = listing_data.get("askPrice0")

    if bid0 is None or ask0 is None:
        return None
    if len(bid0) < MIN_LOOKBACK or len(ask0) < MIN_LOOKBACK:
        return None

    features = np.empty(len(FEATURE_NAMES), dtype=np.float64)

    # Current values
    cur_bid = bid0[-1]
    cur_ask = ask0[-1]
    cur_mid = (cur_bid + cur_ask) / 2.0

    # Helper: get latest value or 0
    def _last(key: str) -> float:
        arr = listing_data.get(key)
        if arr is not None and len(arr) > 0:
            return float(arr[-1])
        return 0.0

    # --- Imbalance ---
    bs0 = _last("bidSize0")
    as0 = _last("askSize0")
    features[0] = _safe_imbalance(bs0, as0)  # tob_imbalance

    def _depth_imb(n: int) -> float:
        b = sum(_last(f"bidSize{i}") for i in range(n))
        a = sum(_last(f"askSize{i}") for i in range(n))
        return _safe_imbalance(b, a)

    features[1] = _depth_imb(3)   # depth3_imbalance
    features[2] = _depth_imb(5)   # depth5_imbalance
    features[3] = _depth_imb(10)  # depth10_imbalance

    bc0 = _last("bidCount0")
    ac0 = _last("askCount0")
    features[4] = _safe_imbalance(bc0, ac0)  # count_imbalance

    pw_bid = sum(_last(f"bidPrice{i}") * _last(f"bidSize{i}") for i in range(10))
    pw_ask = sum(_last(f"askPrice{i}") * _last(f"askSize{i}") for i in range(10))
    features[5] = _safe_imbalance(pw_bid, pw_ask)  # price_weighted_imbalance

    # --- Depth ---
    total_bid = sum(_last(f"bidSize{i}") for i in range(10))
    total_ask = sum(_last(f"askSize{i}") for i in range(10))
    features[6] = total_bid   # total_bid_depth
    features[7] = total_ask   # total_ask_depth
    depth_total = total_bid + total_ask
    features[8] = (total_bid / depth_total) if depth_total != 0 else 0.5  # depth_ratio
    features[9] = total_bid - total_ask  # depth_diff

    # --- Spread / Microprice ---
    spread_val = cur_ask - cur_bid
    features[10] = spread_val  # spread
    features[11] = (spread_val / cur_mid) * 1e4 if cur_mid != 0 else 0.0  # spread_bps

    sz_total = bs0 + as0
    if sz_total > 0:
        microprice = (cur_bid * as0 + cur_ask * bs0) / sz_total
    else:
        microprice = cur_mid
    features[12] = microprice  # microprice
    features[13] = (microprice - cur_mid) / cur_mid if cur_mid != 0 else 0.0  # microprice_deviation

    # --- Momentum ---
    # Build mid-price history for returns/volatility
    n = len(bid0)
    mid_history = (bid0 + ask0) / 2.0
    log_mid = np.log(mid_history)
    log_returns = np.diff(log_mid)  # length n-1

    # Returns: indices 14-17 (return_5, return_10, return_20, return_50)
    windows = [5, 10, 20, 50]
    for i, window in enumerate(windows):
        if n >= window + 1:
            features[14 + i] = log_mid[-1] - log_mid[-(window + 1)]
        else:
            features[14 + i] = np.nan

    # Volatilities: indices 18-21 (volatility_5, volatility_10, volatility_20, volatility_50)
    for i, window in enumerate(windows):
        if len(log_returns) >= window:
            features[18 + i] = float(np.std(log_returns[-window:], ddof=1))
        else:
            features[18 + i] = np.nan

    # --- Dynamics ---
    # imbalance_change_5: tob_imbalance now vs 5 ticks ago
    if n >= 6:
        bs0_prev = float(listing_data.get("bidSize0", np.array([0.0]))[-6]) if len(listing_data.get("bidSize0", np.array([]))) >= 6 else 0.0
        as0_prev = float(listing_data.get("askSize0", np.array([0.0]))[-6]) if len(listing_data.get("askSize0", np.array([]))) >= 6 else 0.0
        imb_prev = _safe_imbalance(bs0_prev, as0_prev)
        features[22] = features[0] - imb_prev  # imbalance_change_5
    else:
        features[22] = np.nan

    # spread_change_5
    if n >= 6:
        prev_bid = float(bid0[-6])
        prev_ask = float(ask0[-6])
        prev_mid = (prev_bid + prev_ask) / 2.0
        prev_spread_bps = ((prev_ask - prev_bid) / prev_mid) * 1e4 if prev_mid != 0 else 0.0
        features[23] = features[11] - prev_spread_bps  # spread_change_5
    else:
        features[23] = np.nan

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
        features[24] = features[9] - prev_depth_diff  # depth_change_5
    else:
        features[24] = np.nan

    # vwap_deviation
    vwap_num = 0.0
    vwap_den = 0.0
    for i in range(5):
        bp = _last(f"bidPrice{i}")
        bs = _last(f"bidSize{i}")
        ap = _last(f"askPrice{i}")
        as_ = _last(f"askSize{i}")
        vwap_num += bp * bs + ap * as_
        vwap_den += bs + as_
    vwap = (vwap_num / vwap_den) if vwap_den > 0 else cur_mid
    features[25] = (cur_mid - vwap) / cur_mid if cur_mid != 0 else 0.0  # vwap_deviation

    # --- Trade flow (book-inferred) ---
    bid_price_arr = listing_data.get("bidPrice0")
    ask_price_arr = listing_data.get("askPrice0")
    bid_size_arr = listing_data.get("bidSize0")
    ask_size_arr = listing_data.get("askSize0")

    if (bid_price_arr is not None and ask_price_arr is not None
            and bid_size_arr is not None and ask_size_arr is not None
            and len(bid_price_arr) >= 21):
        # Infer buy/sell volume from consecutive TOB snapshots
        same_ask = ask_price_arr[1:] == ask_price_arr[:-1]
        ask_dec = np.maximum(ask_size_arr[:-1] - ask_size_arr[1:], 0)
        inferred_buy = np.where(same_ask, ask_dec, 0)

        same_bid = bid_price_arr[1:] == bid_price_arr[:-1]
        bid_dec = np.maximum(bid_size_arr[:-1] - bid_size_arr[1:], 0)
        inferred_sell = np.where(same_bid, bid_dec, 0)

        for i, w in enumerate([5, 20]):
            buy_sum = float(inferred_buy[-w:].sum()) if len(inferred_buy) >= w else float(inferred_buy.sum())
            sell_sum = float(inferred_sell[-w:].sum()) if len(inferred_sell) >= w else float(inferred_sell.sum())
            features[26 + i] = buy_sum      # inferred_buy_vol_{w}
            features[28 + i] = sell_sum      # inferred_sell_vol_{w}
            total_flow = buy_sum + sell_sum
            features[30 + i] = (buy_sum - sell_sum) / total_flow if total_flow != 0 else 0.0  # trade_imbalance_{w}
    else:
        features[26:32] = np.nan

    # --- Z-score normalization (200-tick window) ---
    z_window = 200
    if n >= z_window:
        # tob_imbalance_z (index 32)
        bs_arr = listing_data["bidSize0"][-z_window:]
        as_arr = listing_data["askSize0"][-z_window:]
        tot = bs_arr + as_arr
        imb_hist = np.where(tot == 0, 0, (bs_arr - as_arr) / np.where(tot == 0, 1, tot))
        mean, std = imb_hist.mean(), imb_hist.std(ddof=1)
        features[32] = (imb_hist[-1] - mean) / max(std, 1e-10)

        # depth_ratio_z (index 33)
        bid_depth_hist = np.zeros(z_window)
        ask_depth_hist = np.zeros(z_window)
        for i in range(10):
            b_arr = listing_data.get(f"bidSize{i}")
            a_arr = listing_data.get(f"askSize{i}")
            if b_arr is not None and len(b_arr) >= z_window:
                bid_depth_hist += b_arr[-z_window:]
            if a_arr is not None and len(a_arr) >= z_window:
                ask_depth_hist += a_arr[-z_window:]
        depth_total_hist = bid_depth_hist + ask_depth_hist
        ratio_hist = np.where(depth_total_hist == 0, 0.5, bid_depth_hist / depth_total_hist)
        mean, std = ratio_hist.mean(), ratio_hist.std(ddof=1)
        features[33] = (ratio_hist[-1] - mean) / max(std, 1e-10)

        # total_bid_depth_z (index 34)
        mean, std = bid_depth_hist.mean(), bid_depth_hist.std(ddof=1)
        features[34] = (bid_depth_hist[-1] - mean) / max(std, 1e-10)

        # total_ask_depth_z (index 35)
        mean, std = ask_depth_hist.mean(), ask_depth_hist.std(ddof=1)
        features[35] = (ask_depth_hist[-1] - mean) / max(std, 1e-10)

        # spread_bps_z (index 36)
        bid_p = listing_data["bidPrice0"][-z_window:]
        ask_p = listing_data["askPrice0"][-z_window:]
        mid_hist = (bid_p + ask_p) / 2.0
        spread_hist = np.where(mid_hist == 0, 0, (ask_p - bid_p) / mid_hist * 1e4)
        mean, std = spread_hist.mean(), spread_hist.std(ddof=1)
        features[36] = (spread_hist[-1] - mean) / max(std, 1e-10)

        # depth_diff_z (index 37)
        diff_hist = bid_depth_hist - ask_depth_hist
        mean, std = diff_hist.mean(), diff_hist.std(ddof=1)
        features[37] = (diff_hist[-1] - mean) / max(std, 1e-10)
    else:
        features[32:38] = np.nan

    # --- Depth change decomposition ---
    if n >= 21:
        bid_withdrawal = np.zeros(n - 1)
        ask_withdrawal = np.zeros(n - 1)
        for i in range(10):
            b_arr = listing_data.get(f"bidSize{i}")
            if b_arr is not None and len(b_arr) >= n:
                bid_withdrawal += np.abs(np.minimum(np.diff(b_arr[-n:]), 0))
            a_arr = listing_data.get(f"askSize{i}")
            if a_arr is not None and len(a_arr) >= n:
                ask_withdrawal += np.abs(np.minimum(np.diff(a_arr[-n:]), 0))

        for j, w in enumerate([5, 20]):
            bid_wd = float(bid_withdrawal[-w:].sum()) if len(bid_withdrawal) >= w else float(bid_withdrawal.sum())
            ask_wd = float(ask_withdrawal[-w:].sum()) if len(ask_withdrawal) >= w else float(ask_withdrawal.sum())
            features[38 + j * 3] = bid_wd      # bid_withdrawal_{w}
            features[39 + j * 3] = ask_wd      # ask_withdrawal_{w}
            wd_total = bid_wd + ask_wd
            features[40 + j * 3] = (bid_wd - ask_wd) / wd_total if wd_total != 0 else 0.0  # withdrawal_imbalance_{w}
    else:
        features[38:44] = np.nan

    return features
