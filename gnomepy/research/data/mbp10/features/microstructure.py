"""
Microstructure features for MBP10 numpy-array data.

Covers snapshot-based order book features:
- Imbalance (TOB, depth-N, count, price-weighted)
- Spread (raw + bps + rolling stats)
- Depth (total/ratio/diff/change)
- Microprice and VWAP deviation
- Dynamics (imbalance_change_5, spread_change_5)
- Z-score normalized variants (200-tick window)

Provides two extraction modes:
- compute_microstructure_bulk: vectorized over a DataFrame (for training)
- compute_microstructure_tick: single-observation from numpy arrays (for inference)
"""

import numpy as np
import pandas as pd


def _safe_imbalance(bid_val: float, ask_val: float) -> float:
    total = bid_val + ask_val
    if total == 0:
        return 0.0
    return (bid_val - ask_val) / total


def compute_microstructure_bulk(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all microstructure features over a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: bidPrice0..bidPrice9, askPrice0..askPrice9,
        bidSize0..bidSize9, askSize0..askSize9, bidCount0..bidCount9,
        askCount0..askCount9. Also needs 'midPrice' column or it will be
        computed from bidPrice0/askPrice0.

    Returns
    -------
    pd.DataFrame
        DataFrame with all microstructure feature columns. Rows where features
        cannot be computed (insufficient lookback) will contain NaN.
    """
    out = pd.DataFrame(index=df.index)

    if "midPrice" not in df.columns:
        df = df.copy()
        df["midPrice"] = (df["bidPrice0"] + df["askPrice0"]) / 2.0

    mid = df["midPrice"]

    # ------------------------------------------------------------------
    # Imbalance features
    # ------------------------------------------------------------------
    bid_sz0 = df["bidSize0"]
    ask_sz0 = df["askSize0"]
    out["tob_imbalance"] = (bid_sz0 - ask_sz0) / (bid_sz0 + ask_sz0).replace(0, np.nan)
    out["tob_imbalance"] = out["tob_imbalance"].fillna(0.0)

    def _depth_imbalance(n: int) -> pd.Series:
        bid_sum = sum(df[f"bidSize{i}"] for i in range(n))
        ask_sum = sum(df[f"askSize{i}"] for i in range(n))
        total = bid_sum + ask_sum
        return ((bid_sum - ask_sum) / total.replace(0, np.nan)).fillna(0.0)

    out["depth3_imbalance"] = _depth_imbalance(3)
    out["depth5_imbalance"] = _depth_imbalance(5)
    out["depth10_imbalance"] = _depth_imbalance(10)

    bid_ct0 = df.get("bidCount0", pd.Series(0, index=df.index))
    ask_ct0 = df.get("askCount0", pd.Series(0, index=df.index))
    ct_total = bid_ct0 + ask_ct0
    out["count_imbalance"] = ((bid_ct0 - ask_ct0) / ct_total.replace(0, np.nan)).fillna(0.0)

    pw_bid = sum(df[f"bidPrice{i}"] * df[f"bidSize{i}"] for i in range(10))
    pw_ask = sum(df[f"askPrice{i}"] * df[f"askSize{i}"] for i in range(10))
    pw_total = pw_bid + pw_ask
    out["price_weighted_imbalance"] = ((pw_bid - pw_ask) / pw_total.replace(0, np.nan)).fillna(0.0)

    # ------------------------------------------------------------------
    # Depth features
    # ------------------------------------------------------------------
    total_bid: pd.Series = sum(df[f"bidSize{i}"] for i in range(10))  # type: ignore[assignment]
    total_ask: pd.Series = sum(df[f"askSize{i}"] for i in range(10))  # type: ignore[assignment]
    depth_total = total_bid + total_ask
    out["total_bid_depth"] = total_bid
    out["total_ask_depth"] = total_ask
    out["depth_ratio"] = (total_bid / depth_total.replace(0, np.nan)).fillna(0.5)
    out["depth_diff"] = total_bid - total_ask

    # ------------------------------------------------------------------
    # Spread features
    # ------------------------------------------------------------------
    spread = df["askPrice0"] - df["bidPrice0"]
    spread_bps = (spread / mid) * 1e4
    out["spread"] = spread
    out["spread_bps"] = spread_bps
    out["spread_mean_20"] = spread_bps.rolling(20).mean()
    out["spread_std_10"] = spread_bps.rolling(10).std()

    # ------------------------------------------------------------------
    # Microprice and VWAP deviation
    # ------------------------------------------------------------------
    sz_total = bid_sz0 + ask_sz0
    microprice = (df["bidPrice0"] * ask_sz0 + df["askPrice0"] * bid_sz0) / sz_total.replace(0, np.nan)
    microprice = microprice.fillna(mid)
    out["microprice"] = microprice
    out["microprice_deviation"] = (microprice - mid) / mid

    vwap_num = sum(df[f"bidPrice{i}"] * df[f"bidSize{i}"] + df[f"askPrice{i}"] * df[f"askSize{i}"] for i in range(5))
    vwap_den = sum(df[f"bidSize{i}"] + df[f"askSize{i}"] for i in range(5))
    vwap = (vwap_num / vwap_den.replace(0, np.nan)).fillna(mid)
    out["vwap_deviation"] = (mid - vwap) / mid

    # ------------------------------------------------------------------
    # Dynamics (changes over 5 ticks)
    # ------------------------------------------------------------------
    out["depth_change_5"] = (total_bid - total_ask).diff(5)
    out["imbalance_change_5"] = out["tob_imbalance"].diff(5)
    out["spread_change_5"] = out["spread_bps"].diff(5)

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

    return out


def compute_microstructure_tick(listing_data: dict[str, np.ndarray]) -> dict[str, float]:
    """Extract all microstructure features for the most recent tick.

    Parameters
    ----------
    listing_data : dict[str, np.ndarray]
        Dict mapping column names to numpy arrays of historical values.

    Returns
    -------
    dict[str, float]
        Feature name -> value mapping for all microstructure features.
    """
    bid0 = listing_data.get("bidPrice0")
    ask0 = listing_data.get("askPrice0")

    out: dict[str, float] = {}

    def _last(key: str) -> float:
        arr = listing_data.get(key)
        if arr is not None and len(arr) > 0:
            return float(arr[-1])
        return 0.0

    cur_bid = _last("bidPrice0")
    cur_ask = _last("askPrice0")
    cur_mid = (cur_bid + cur_ask) / 2.0
    n = len(bid0) if bid0 is not None else 0

    # --- Imbalance ---
    bs0 = _last("bidSize0")
    as0 = _last("askSize0")
    out["tob_imbalance"] = _safe_imbalance(bs0, as0)

    def _depth_imb(levels: int) -> float:
        b = sum(_last(f"bidSize{i}") for i in range(levels))
        a = sum(_last(f"askSize{i}") for i in range(levels))
        return _safe_imbalance(b, a)

    out["depth3_imbalance"] = _depth_imb(3)
    out["depth5_imbalance"] = _depth_imb(5)
    out["depth10_imbalance"] = _depth_imb(10)

    bc0 = _last("bidCount0")
    ac0 = _last("askCount0")
    out["count_imbalance"] = _safe_imbalance(bc0, ac0)

    pw_bid = sum(_last(f"bidPrice{i}") * _last(f"bidSize{i}") for i in range(10))
    pw_ask = sum(_last(f"askPrice{i}") * _last(f"askSize{i}") for i in range(10))
    out["price_weighted_imbalance"] = _safe_imbalance(pw_bid, pw_ask)

    # --- Depth ---
    total_bid = sum(_last(f"bidSize{i}") for i in range(10))
    total_ask = sum(_last(f"askSize{i}") for i in range(10))
    depth_total = total_bid + total_ask
    out["total_bid_depth"] = total_bid
    out["total_ask_depth"] = total_ask
    out["depth_ratio"] = (total_bid / depth_total) if depth_total != 0 else 0.5
    out["depth_diff"] = total_bid - total_ask

    # --- Spread ---
    spread_val = cur_ask - cur_bid
    out["spread"] = spread_val
    out["spread_bps"] = (spread_val / cur_mid) * 1e4 if cur_mid != 0 else 0.0

    if n >= 20 and bid0 is not None and ask0 is not None:
        spread_hist = ask0[-20:] - bid0[-20:]
        mid_hist = (bid0[-20:] + ask0[-20:]) / 2.0
        spread_bps_hist = np.where(mid_hist == 0, 0, spread_hist / mid_hist * 1e4)
        out["spread_mean_20"] = float(np.mean(spread_bps_hist))
    else:
        out["spread_mean_20"] = out["spread_bps"]

    if n >= 10 and bid0 is not None and ask0 is not None:
        spread_hist_10 = ask0[-10:] - bid0[-10:]
        mid_hist_10 = (bid0[-10:] + ask0[-10:]) / 2.0
        spread_bps_10 = np.where(mid_hist_10 == 0, 0, spread_hist_10 / mid_hist_10 * 1e4)
        out["spread_std_10"] = float(np.std(spread_bps_10, ddof=1))
    else:
        out["spread_std_10"] = np.nan

    # --- Microprice and VWAP ---
    sz_total = bs0 + as0
    if sz_total > 0:
        microprice = (cur_bid * as0 + cur_ask * bs0) / sz_total
    else:
        microprice = cur_mid
    out["microprice"] = microprice
    out["microprice_deviation"] = (microprice - cur_mid) / cur_mid if cur_mid != 0 else 0.0

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
    out["vwap_deviation"] = (cur_mid - vwap) / cur_mid if cur_mid != 0 else 0.0

    # --- Dynamics ---
    if n >= 6 and bid0 is not None and ask0 is not None:
        prev_depth_diff = 0.0
        for i in range(10):
            b_arr = listing_data.get(f"bidSize{i}")
            a_arr = listing_data.get(f"askSize{i}")
            if b_arr is not None and len(b_arr) >= 6:
                prev_depth_diff += float(b_arr[-6])
            if a_arr is not None and len(a_arr) >= 6:
                prev_depth_diff -= float(a_arr[-6])
        out["depth_change_5"] = out["depth_diff"] - prev_depth_diff

        bs0_prev = float(listing_data.get("bidSize0", np.array([0.0]))[-6]) if len(listing_data.get("bidSize0", np.array([]))) >= 6 else 0.0
        as0_prev = float(listing_data.get("askSize0", np.array([0.0]))[-6]) if len(listing_data.get("askSize0", np.array([]))) >= 6 else 0.0
        imb_prev = _safe_imbalance(bs0_prev, as0_prev)
        out["imbalance_change_5"] = out["tob_imbalance"] - imb_prev

        prev_bid = float(bid0[-6])
        prev_ask = float(ask0[-6])
        prev_mid = (prev_bid + prev_ask) / 2.0
        prev_spread_bps = ((prev_ask - prev_bid) / prev_mid) * 1e4 if prev_mid != 0 else 0.0
        out["spread_change_5"] = out["spread_bps"] - prev_spread_bps
    else:
        out["depth_change_5"] = np.nan
        out["imbalance_change_5"] = np.nan
        out["spread_change_5"] = np.nan

    # --- Z-score normalization (200-tick window) ---
    z_window = 200
    if n >= z_window and bid0 is not None and ask0 is not None:
        bs_arr = listing_data["bidSize0"][-z_window:]
        as_arr = listing_data["askSize0"][-z_window:]
        tot = bs_arr + as_arr
        imb_hist = np.where(tot == 0, 0, (bs_arr - as_arr) / np.where(tot == 0, 1, tot))
        mean, std = imb_hist.mean(), imb_hist.std(ddof=1)
        out["tob_imbalance_z"] = (imb_hist[-1] - mean) / max(std, 1e-10)

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
        out["depth_ratio_z"] = (ratio_hist[-1] - mean) / max(std, 1e-10)

        mean, std = bid_depth_hist.mean(), bid_depth_hist.std(ddof=1)
        out["total_bid_depth_z"] = (bid_depth_hist[-1] - mean) / max(std, 1e-10)

        mean, std = ask_depth_hist.mean(), ask_depth_hist.std(ddof=1)
        out["total_ask_depth_z"] = (ask_depth_hist[-1] - mean) / max(std, 1e-10)

        bid_p = listing_data["bidPrice0"][-z_window:]
        ask_p = listing_data["askPrice0"][-z_window:]
        mid_hist = (bid_p + ask_p) / 2.0
        spread_hist = np.where(mid_hist == 0, 0, (ask_p - bid_p) / mid_hist * 1e4)
        mean, std = spread_hist.mean(), spread_hist.std(ddof=1)
        out["spread_bps_z"] = (spread_hist[-1] - mean) / max(std, 1e-10)

        diff_hist = bid_depth_hist - ask_depth_hist
        mean, std = diff_hist.mean(), diff_hist.std(ddof=1)
        out["depth_diff_z"] = (diff_hist[-1] - mean) / max(std, 1e-10)
    else:
        out["tob_imbalance_z"] = np.nan
        out["depth_ratio_z"] = np.nan
        out["total_bid_depth_z"] = np.nan
        out["total_ask_depth_z"] = np.nan
        out["spread_bps_z"] = np.nan
        out["depth_diff_z"] = np.nan

    return out
