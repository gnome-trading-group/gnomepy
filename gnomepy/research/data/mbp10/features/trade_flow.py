"""
Trade flow features for MBP10 numpy-array data.

Covers book-inferred trade activity and depth withdrawals:
- Inferred buy/sell volume from consecutive TOB snapshots
- Trade imbalance at windows 5 and 20
- Bid/ask withdrawal sums at windows 5 and 20
- Withdrawal imbalance at windows 5 and 20

Provides two extraction modes:
- compute_trade_flow_bulk: vectorized over a DataFrame (for training)
- compute_trade_flow_tick: single-observation from numpy arrays (for inference)
"""

import numpy as np
import pandas as pd


def compute_trade_flow_bulk(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all trade flow features over a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: bidPrice0, askPrice0, bidSize0..bidSize9,
        askSize0..askSize9.

    Returns
    -------
    pd.DataFrame
        DataFrame with all trade flow feature columns. Rows where features
        cannot be computed (insufficient lookback) will contain NaN.
    """
    out = pd.DataFrame(index=df.index)

    # ------------------------------------------------------------------
    # Inferred buy/sell volume from consecutive TOB snapshots
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
    # Depth withdrawals
    # ------------------------------------------------------------------
    for side, prefix in [("bid", "bid"), ("ask", "ask")]:
        total_withdrawal: pd.Series = sum(  # type: ignore[assignment]
            df[f"{prefix}Size{i}"].diff(1).clip(upper=0) for i in range(10)
        ).abs()  # type: ignore[union-attr]
        for w in [5, 20]:
            out[f"{side}_withdrawal_{w}"] = total_withdrawal.rolling(w).sum()

    for w in [5, 20]:
        wd_total = out[f"bid_withdrawal_{w}"] + out[f"ask_withdrawal_{w}"]
        out[f"withdrawal_imbalance_{w}"] = (
            (out[f"bid_withdrawal_{w}"] - out[f"ask_withdrawal_{w}"])
            / wd_total.replace(0, np.nan)
        ).fillna(0)

    return out


def compute_trade_flow_tick(listing_data: dict[str, np.ndarray]) -> dict[str, float]:
    """Extract all trade flow features for the most recent tick.

    Parameters
    ----------
    listing_data : dict[str, np.ndarray]
        Dict mapping column names to numpy arrays of historical values.

    Returns
    -------
    dict[str, float]
        Feature name -> value mapping for all trade flow features.
    """
    out: dict[str, float] = {}

    bid_price_arr = listing_data.get("bidPrice0")
    ask_price_arr = listing_data.get("askPrice0")
    bid_size_arr = listing_data.get("bidSize0")
    ask_size_arr = listing_data.get("askSize0")

    n = len(bid_price_arr) if bid_price_arr is not None else 0

    # --- Inferred trade flow ---
    if (bid_price_arr is not None and ask_price_arr is not None
            and bid_size_arr is not None and ask_size_arr is not None
            and len(bid_price_arr) >= 21):
        same_ask = ask_price_arr[1:] == ask_price_arr[:-1]
        ask_dec = np.maximum(ask_size_arr[:-1] - ask_size_arr[1:], 0)
        inferred_buy = np.where(same_ask, ask_dec, 0)

        same_bid = bid_price_arr[1:] == bid_price_arr[:-1]
        bid_dec = np.maximum(bid_size_arr[:-1] - bid_size_arr[1:], 0)
        inferred_sell = np.where(same_bid, bid_dec, 0)

        for w in [5, 20]:
            buy_sum = float(inferred_buy[-w:].sum()) if len(inferred_buy) >= w else float(inferred_buy.sum())
            sell_sum = float(inferred_sell[-w:].sum()) if len(inferred_sell) >= w else float(inferred_sell.sum())
            out[f"inferred_buy_vol_{w}"] = buy_sum
            out[f"inferred_sell_vol_{w}"] = sell_sum
            total_flow = buy_sum + sell_sum
            out[f"trade_imbalance_{w}"] = (buy_sum - sell_sum) / total_flow if total_flow != 0 else 0.0
    else:
        for w in [5, 20]:
            out[f"inferred_buy_vol_{w}"] = np.nan
            out[f"inferred_sell_vol_{w}"] = np.nan
            out[f"trade_imbalance_{w}"] = np.nan

    # --- Depth withdrawals ---
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

        for w in [5, 20]:
            bid_wd = float(bid_withdrawal[-w:].sum()) if len(bid_withdrawal) >= w else float(bid_withdrawal.sum())
            ask_wd = float(ask_withdrawal[-w:].sum()) if len(ask_withdrawal) >= w else float(ask_withdrawal.sum())
            out[f"bid_withdrawal_{w}"] = bid_wd
            out[f"ask_withdrawal_{w}"] = ask_wd
            wd_total = bid_wd + ask_wd
            out[f"withdrawal_imbalance_{w}"] = (bid_wd - ask_wd) / wd_total if wd_total != 0 else 0.0
    else:
        for w in [5, 20]:
            out[f"bid_withdrawal_{w}"] = np.nan
            out[f"ask_withdrawal_{w}"] = np.nan
            out[f"withdrawal_imbalance_{w}"] = np.nan

    return out
