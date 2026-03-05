from __future__ import annotations

import pandas as pd


def _get_mpl():
    import matplotlib
    import matplotlib.pyplot as plt

    return matplotlib, plt


def plot_spread(df: pd.DataFrame, bps: bool = True, rolling_window: int = 100, ax=None):
    _, plt = _get_mpl()
    col = "spread_bps" if bps else "spread"
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))
    else:
        fig = ax.get_figure()
    ax.plot(df.index, df[col], alpha=0.3, linewidth=0.5, label=col)
    ax.plot(df.index, df[col].rolling(rolling_window).mean(), color="red", label=f"rolling {rolling_window}")
    ax.set_ylabel("Spread (bps)" if bps else "Spread")
    ax.set_title("Spread Over Time")
    ax.legend()
    return fig


def plot_depth_profile(df: pd.DataFrame, at_index=None, n_levels: int = 10, ax=None):
    import numpy as np

    _, plt = _get_mpl()
    if at_index is None:
        at_index = len(df) // 2
    row = df.iloc[at_index]

    levels = list(range(n_levels))
    bid_sizes = [row.get(f"bidSize{i}", 0) for i in levels]
    ask_sizes = [row.get(f"askSize{i}", 0) for i in levels]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    y = np.arange(n_levels)
    ax.barh(y + 0.15, bid_sizes, height=0.3, label="Bid", color="green", alpha=0.7)
    ax.barh(y - 0.15, ask_sizes, height=0.3, label="Ask", color="red", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"Level {i}" for i in levels])
    ax.set_xlabel("Size")
    ax.set_title(f"Depth Profile (index={at_index})")
    ax.legend()
    return fig


def plot_imbalance(df: pd.DataFrame, metrics: list[str] | None = None, rolling_window: int = 50, ax=None):
    _, plt = _get_mpl()
    if metrics is None:
        candidates = ["tob_imbalance", "depth5_imbalance", "depth10_imbalance"]
        metrics = [c for c in candidates if c in df.columns]
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))
    else:
        fig = ax.get_figure()
    for col in metrics:
        ax.plot(df.index, df[col].rolling(rolling_window).mean(), label=col, alpha=0.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_ylabel("Imbalance")
    ax.set_title("Order Book Imbalance")
    ax.legend()
    return fig


def plot_microprice(df: pd.DataFrame, ax=None):
    _, plt = _get_mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))
    else:
        fig = ax.get_figure()
    ax.plot(df.index, df["midPrice"], label="midPrice", alpha=0.7)
    ax.plot(df.index, df["microprice"], label="microprice", alpha=0.7)
    ax.set_ylabel("Price")
    ax.set_title("Mid-Price vs Microprice")
    ax.legend()
    return fig


def plot_trade_flow(df: pd.DataFrame, window: int = 20, ax=None):
    _, plt = _get_mpl()
    buy_col = f"inferred_buy_vol_{window}"
    sell_col = f"inferred_sell_vol_{window}"
    imb_col = f"trade_imbalance_{window}"

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))
    else:
        fig = ax.get_figure()

    if buy_col in df.columns and sell_col in df.columns:
        ax.fill_between(df.index, df[buy_col], alpha=0.4, color="green", label="Buy vol")
        ax.fill_between(df.index, df[sell_col], alpha=0.4, color="red", label="Sell vol")
        ax.set_ylabel("Volume")

    if imb_col in df.columns:
        ax2 = ax.twinx()
        ax2.plot(df.index, df[imb_col], color="blue", alpha=0.6, linewidth=0.8, label="Trade imbalance")
        ax2.set_ylabel("Imbalance")
        ax2.legend(loc="upper left")

    ax.set_title("Trade Flow")
    ax.legend(loc="upper right")
    return fig


def plot_book_heatmap(df: pd.DataFrame, levels: int = 5, max_rows: int = 500, ax=None):
    import numpy as np

    _, plt = _get_mpl()
    sub = df.iloc[:max_rows]

    bid_cols = [f"bidSize{i}" for i in range(levels)]
    ask_cols = [f"askSize{i}" for i in range(levels)]
    data = np.column_stack([
        sub[bid_cols].values[:, ::-1],
        sub[ask_cols].values,
    ])

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    im = ax.imshow(data.T, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_ylabel("Level")
    ax.set_xlabel("Time index")
    labels = [f"B{i}" for i in range(levels - 1, -1, -1)] + [f"A{i}" for i in range(levels)]
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Book Depth Heatmap")
    fig.colorbar(im, ax=ax, label="Size")
    return fig


def plot_dashboard(df: pd.DataFrame, figsize=(16, 12)):
    _, plt = _get_mpl()
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    plot_spread(df, ax=axes[0, 0])
    plot_imbalance(df, ax=axes[0, 1])
    plot_microprice(df, ax=axes[1, 0])
    plot_depth_profile(df, ax=axes[1, 1])
    fig.tight_layout()
    return fig
