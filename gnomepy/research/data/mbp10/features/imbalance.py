"""
Order book imbalance calculation functions for MBP10 data.

This module provides functions to calculate various order book imbalance metrics
from MBP10 (Market By Price 10-level) data.
"""

from typing import Optional
from gnomepy.data.types import MBP10, BidAskPair


def calculate_order_book_imbalance(
    mbp10_data: MBP10,
    levels: Optional[int] = None,
    method: str = "volume_weighted"
) -> float:
    """
    Calculate order book imbalance from MBP10 data.

    The order book imbalance measures the relative strength of buy vs sell pressure
    in the order book. It ranges from -1 (all sell pressure) to +1 (all buy pressure).

    Args:
        mbp10_data: MBP10 market data containing order book levels
        levels: Number of levels to include in calculation (default: all available)
        method: Calculation method - "volume_weighted", "simple", or "price_weighted"

    Returns:
        float: Order book imbalance value between -1 and 1

    Raises:
        ValueError: If method is not supported or levels is invalid
    """
    if not mbp10_data.levels:
        return 0.0

    # Filter out levels with zero prices (empty levels)
    valid_levels = [level for level in mbp10_data.levels
                   if level.bid_px > 0 and level.ask_px > 0]

    if not valid_levels:
        return 0.0

    # Limit to specified number of levels
    if levels is not None:
        valid_levels = valid_levels[:levels]

    if method == "volume_weighted":
        return _calculate_volume_weighted_imbalance(valid_levels)
    elif method == "simple":
        return _calculate_simple_imbalance(valid_levels)
    elif method == "price_weighted":
        return _calculate_price_weighted_imbalance(valid_levels)
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'volume_weighted', 'simple', or 'price_weighted'")


def _calculate_volume_weighted_imbalance(levels: list[BidAskPair]) -> float:
    """
    Calculate volume-weighted order book imbalance.

    This method weights each level by its volume (size) and calculates the imbalance
    as (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume).
    """
    total_bid_volume = sum(level.bid_sz for level in levels)
    total_ask_volume = sum(level.ask_sz for level in levels)

    if total_bid_volume + total_ask_volume == 0:
        return 0.0

    return (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)


def _calculate_simple_imbalance(levels: list[BidAskPair]) -> float:
    """
    Calculate simple order book imbalance.

    This method treats each level equally and calculates the imbalance
    as (bid_levels - ask_levels) / (bid_levels + ask_levels).
    """
    bid_levels = sum(1 for level in levels if level.bid_sz > 0)
    ask_levels = sum(1 for level in levels if level.ask_sz > 0)

    if bid_levels + ask_levels == 0:
        return 0.0

    return (bid_levels - ask_levels) / (bid_levels + ask_levels)


def _calculate_price_weighted_imbalance(levels: list[BidAskPair]) -> float:
    """
    Calculate price-weighted order book imbalance.

    This method weights each level by its price and calculates the imbalance
    as (weighted_bid_volume - weighted_ask_volume) / (weighted_bid_volume + weighted_ask_volume).
    """
    weighted_bid_volume = sum(level.bid_px * level.bid_sz for level in levels)
    weighted_ask_volume = sum(level.ask_px * level.ask_sz for level in levels)

    if weighted_bid_volume + weighted_ask_volume == 0:
        return 0.0

    return (weighted_bid_volume - weighted_ask_volume) / (weighted_bid_volume + weighted_ask_volume)


def calculate_top_of_book_imbalance(mbp10_data: MBP10) -> float:
    """
    Calculate order book imbalance using only the top of book (best bid/ask).

    Args:
        mbp10_data: MBP10 market data containing order book levels

    Returns:
        float: Top of book imbalance value between -1 and 1
    """
    if not mbp10_data.levels or len(mbp10_data.levels) == 0:
        return 0.0

    top_level = mbp10_data.levels[0]

    if top_level.bid_px <= 0 or top_level.ask_px <= 0:
        return 0.0

    if top_level.bid_sz + top_level.ask_sz == 0:
        return 0.0

    return (top_level.bid_sz - top_level.ask_sz) / (top_level.bid_sz + top_level.ask_sz)


def calculate_mid_price_imbalance(mbp10_data: MBP10, levels: int = 5) -> float:
    """
    Calculate order book imbalance using mid-price weighted approach.

    This method calculates the imbalance by weighting each level by its distance
    from the mid-price, giving more weight to levels closer to the mid-price.

    Args:
        mbp10_data: MBP10 market data containing order book levels
        levels: Number of levels to include in calculation

    Returns:
        float: Mid-price weighted imbalance value between -1 and 1
    """
    if not mbp10_data.levels:
        return 0.0

    valid_levels = [level for level in mbp10_data.levels[:levels]
                   if level.bid_px > 0 and level.ask_px > 0]

    if not valid_levels:
        return 0.0

    # Calculate mid-price from top of book
    top_level = valid_levels[0]
    mid_price = (top_level.bid_px + top_level.ask_px) / 2

    weighted_bid_volume = 0
    weighted_ask_volume = 0

    for level in valid_levels:
        # Calculate weight based on distance from mid-price
        bid_distance = abs(level.bid_px - mid_price) / mid_price
        ask_distance = abs(level.ask_px - mid_price) / mid_price

        # Higher weight for levels closer to mid-price
        bid_weight = 1 / (1 + bid_distance)
        ask_weight = 1 / (1 + ask_distance)

        weighted_bid_volume += level.bid_sz * bid_weight
        weighted_ask_volume += level.ask_sz * ask_weight

    if weighted_bid_volume + weighted_ask_volume == 0:
        return 0.0

    return (weighted_bid_volume - weighted_ask_volume) / (weighted_bid_volume + weighted_ask_volume)
