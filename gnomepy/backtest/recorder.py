from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np

from gnomepy import SchemaBase
from gnomepy.data.types import (
    MBP10, MBP1, MBO, BBO, Trades, OHLCV, 
    FIXED_PRICE_SCALE, FIXED_SIZE_SCALE, SchemaType
)

record_dtype = np.dtype(
    [
        ('event', 'i8'),
        ('timestamp', 'i8'),
        ('price', 'f8'),
        ('quantity', 'f8'),
        ('fee', 'f8'),
    ],
    align=True
)

class RecordType(IntEnum):
    """Enumeration of supported record/event types."""
    MARKET = 1
    EXECUTION = 2


class Recorder:
    """In-memory recorder for market and execution events across multiple assets.

    Events are appended into a preallocated structured NumPy array for speed and
    minimal overhead during backtests.

    Parameters
    ----------
    listing_ids : list[int]
        Unique listing identifiers for tracked assets.
    schema_type : SchemaType
        The schema type being used for market data logging.
    size : int, default 100_000
        Maximum number of events to record per asset.
    auto_resize : bool, default True
        Whether to automatically resize the buffer when it's full.
    """

    def __init__(self, listing_ids: list[int], schema_type: SchemaType, size: int = 100_000, auto_resize: bool = True):
        num_assets = len(listing_ids)
        self.records = np.zeros((size, num_assets), record_dtype)
        self.at_i = np.zeros((num_assets,), dtype=np.int64)
        self.listing_id_to_asset_no = {v: k for k, v in enumerate(listing_ids)}
        self.schema_type = schema_type
        self.auto_resize = auto_resize
        self.initial_size = size

    def log(
        self,
        event: RecordType,
        listing_id: int,
        timestamp: int,
        price: float = None,
        quantity: float = None,
        fee: float = None,
    ):
        """Append an arbitrary event to the record buffer.

        Parameters
        ----------
        event : RecordType
            The event type (e.g., MARKET, EXECUTION).
        listing_id : int
            Asset listing identifier.
        timestamp : int
            Event timestamp (ns or ms consistent with data source).
        price : float, optional
            Event price, if applicable.
        quantity : float, optional
            Position quantity after the event or event quantity.
        fee : float, optional
            Fee associated with the event.

        Raises
        ------
        KeyError
            If listing_id is not found in the recorder.
        IndexError
            If the per-asset buffer is full and auto_resize is disabled.
        ValueError
            If timestamp is negative or invalid.
        """
        # Input validation
        if listing_id not in self.listing_id_to_asset_no:
            raise KeyError(f"Listing ID {listing_id} not found in recorder. Available: {list(self.listing_id_to_asset_no.keys())}")
        
        if timestamp < 0:
            raise ValueError(f"Timestamp must be non-negative, got {timestamp}")
        
        if price is not None and price < 0:
            raise ValueError(f"Price must be non-negative, got {price}")
        
        if fee is not None and fee < 0:
            raise ValueError(f"Fee must be non-negative, got {fee}")

        asset_no = self.listing_id_to_asset_no[listing_id]
        i = self.at_i[asset_no]

        if i >= len(self.records):
            if self.auto_resize:
                self._resize_buffer()
            else:
                raise IndexError(f"Buffer full for asset {listing_id}. Consider increasing size or enabling auto_resize.")

        self.records[i, asset_no]['event'] = event.value
        self.records[i, asset_no]['timestamp'] = timestamp
        self.records[i, asset_no]['price'] = price
        self.records[i, asset_no]['quantity'] = quantity
        self.records[i, asset_no]['fee'] = fee

        self.at_i[asset_no] += 1

    def log_market_event(
        self,
        listing_id: int,
        timestamp: int,
        market_update: SchemaBase,
        quantity: float,
    ):
        """Append a market event derived from a schema update.

        Parameters
        ----------
        listing_id : int
            Asset listing identifier.
        timestamp : int
            Event timestamp (ns or ms consistent with data source).
        market_update : SchemaBase
            A market data update (e.g., `MBP10`, `MBP1`, `MBO`, `BBO`, `Trades`, `OHLCV`).
        quantity : float
            Position quantity after applying the strategy's action at this tick.

        Raises
        ------
        KeyError
            If listing_id is not found in the recorder.
        IndexError
            If the per-asset buffer is full and auto_resize is disabled.
        ValueError
            If `market_update` is not a supported schema type or invalid inputs.
        """
        # Input validation
        if listing_id not in self.listing_id_to_asset_no:
            raise KeyError(f"Listing ID {listing_id} not found in recorder. Available: {list(self.listing_id_to_asset_no.keys())}")
        
        if timestamp < 0:
            raise ValueError(f"Timestamp must be non-negative, got {timestamp}")
        
        if market_update is None:
            raise ValueError("market_update cannot be None")
        
        if not isinstance(market_update, SchemaBase):
            raise ValueError(f"market_update must be a SchemaBase instance, got {type(market_update)}")

        asset_no = self.listing_id_to_asset_no[listing_id]
        i = self.at_i[asset_no]

        if i >= len(self.records):
            if self.auto_resize:
                self._resize_buffer()
            else:
                raise IndexError(f"Buffer full for asset {listing_id}. Consider increasing size or enabling auto_resize.")

        # Extract price based on schema type
        price = self._extract_price_from_schema(market_update)

        self.records[i, asset_no]['event'] = RecordType.MARKET.value
        self.records[i, asset_no]['timestamp'] = timestamp
        self.records[i, asset_no]['price'] = price
        self.records[i, asset_no]['quantity'] = quantity
        self.records[i, asset_no]['fee'] = 0

        self.at_i[asset_no] += 1

    def _extract_price_from_schema(self, market_update: SchemaBase) -> float | None:
        """Extract price from market update based on schema type.
        
        Parameters
        ----------
        market_update : SchemaBase
            The market data update to extract price from.
            
        Returns
        -------
        float | None
            The extracted price, or None if not available.
        """
        if isinstance(market_update, (MBP10, MBP1)):
            # For MBP schemas, use the top of book (first level) mid price
            if market_update.levels and len(market_update.levels) > 0:
                bid_ask_pr = market_update.levels[0]
                if bid_ask_pr.bid_px > 0 and bid_ask_pr.ask_px > 0:
                    return (bid_ask_pr.bid_px + bid_ask_pr.ask_px) / (2.0 * FIXED_PRICE_SCALE)
            return None
            
        elif isinstance(market_update, MBO):
            # For MBO, use the order price directly
            if market_update.price is not None:
                return market_update.price / FIXED_PRICE_SCALE
            return None
            
        elif isinstance(market_update, BBO):
            # For BBO, use the top of book mid price
            if market_update.levels and len(market_update.levels) > 0:
                bid_ask_pr = market_update.levels[0]
                if bid_ask_pr.bid_px > 0 and bid_ask_pr.ask_px > 0:
                    return (bid_ask_pr.bid_px + bid_ask_pr.ask_px) / (2.0 * FIXED_PRICE_SCALE)
            return None
            
        elif isinstance(market_update, Trades):
            # For trades, use the trade price
            if market_update.price is not None:
                return market_update.price / FIXED_PRICE_SCALE
            return None
            
        elif isinstance(market_update, OHLCV):
            # For OHLCV, use the close price
            return market_update.close / FIXED_PRICE_SCALE
            
        else:
            raise ValueError(f"Unsupported schema type: {type(market_update)}")

    def _resize_buffer(self):
        """Dynamically resize the buffer when it's full."""
        old_size = len(self.records)
        new_size = old_size * 2  # Double the size
        
        # Create new larger buffer
        new_records = np.zeros((new_size, self.records.shape[1]), record_dtype)
        
        # Copy existing data
        new_records[:old_size] = self.records
        
        # Replace the old buffer
        self.records = new_records

    def get_buffer_usage(self) -> dict[int, float]:
        """Get buffer usage statistics for each asset.
        
        Returns
        -------
        dict[int, float]
            Dictionary mapping listing_id to usage percentage (0.0 to 1.0).
        """
        usage = {}
        for listing_id, asset_no in self.listing_id_to_asset_no.items():
            usage[listing_id] = self.at_i[asset_no] / len(self.records)
        return usage

    def clear(self):
        """Clear all recorded data and reset counters."""
        self.records.fill(0)
        self.at_i.fill(0)

    def get_record(self, listing_id: int):
        """Return non-empty rows for a given `listing_id` as a structured array."""
        if listing_id not in self.listing_id_to_asset_no:
            raise KeyError(f"Listing ID {listing_id} not found in recorder")

        # Import here to avoid circular imports
        from gnomepy.backtest.stats.stats import Record

        asset_no = self.listing_id_to_asset_no[listing_id]
        used_count = self.at_i[asset_no]

        if used_count == 0:
            data = np.array([], dtype=record_dtype)
        else:
            data = self.records[:used_count, asset_no]

        return Record(arr=data)

    def get_all_records(self):
        """Get records for all assets efficiently.
        
        Returns
        -------
        dict[int, Record]
            Dictionary mapping listing_id to Record objects.
        """
        records = {}
        for listing_id in self.listing_id_to_asset_no.keys():
            records[listing_id] = self.get_record(listing_id)
        return records

    def get_record_count(self, listing_id: int) -> int:
        """Get the number of records for a specific asset.
        
        Parameters
        ----------
        listing_id : int
            Asset listing identifier.
            
        Returns
        -------
        int
            Number of records for the asset.
        """
        if listing_id not in self.listing_id_to_asset_no:
            raise KeyError(f"Listing ID {listing_id} not found in recorder")
        
        asset_no = self.listing_id_to_asset_no[listing_id]
        return int(self.at_i[asset_no])

    def get_total_record_count(self) -> int:
        """Get total number of records across all assets.
        
        Returns
        -------
        int
            Total number of records.
        """
        return int(np.sum(self.at_i))

    def validate_data_integrity(self) -> dict[str, Any]:
        """Validate data integrity and consistency across all records.
        
        Returns
        -------
        dict[str, Any]
            Dictionary containing validation results and any issues found.
        """
        issues = []
        stats = {
            'total_records': self.get_total_record_count(),
            'buffer_usage': self.get_buffer_usage(),
            'schema_type': self.schema_type.value,
            'issues': issues
        }
        
        # Check for timestamp ordering issues
        for listing_id in self.listing_id_to_asset_no.keys():
            record = self.get_record(listing_id)
            if len(record.arr) > 1:
                timestamps = record.arr['timestamp']
                if not np.all(timestamps[:-1] <= timestamps[1:]):
                    issues.append(f"Non-monotonic timestamps detected for asset {listing_id}")
        
        # Check for negative prices
        for listing_id in self.listing_id_to_asset_no.keys():
            record = self.get_record(listing_id)
            if len(record.arr) > 0:
                prices = record.arr['price']
                negative_prices = np.sum(prices < 0)
                if negative_prices > 0:
                    issues.append(f"Found {negative_prices} negative prices for asset {listing_id}")
        
        # Check for negative fees
        for listing_id in self.listing_id_to_asset_no.keys():
            record = self.get_record(listing_id)
            if len(record.arr) > 0:
                fees = record.arr['fee']
                negative_fees = np.sum(fees < 0)
                if negative_fees > 0:
                    issues.append(f"Found {negative_fees} negative fees for asset {listing_id}")
        
        stats['is_valid'] = len(issues) == 0
        return stats

    def get_summary_stats(self) -> dict[str, Any]:
        """Get statistics for all recorded data.
        
        Returns
        -------
        dict[str, Any]
            Dictionary containing summary statistics.
        """
        summary = {
            'total_records': self.get_total_record_count(),
            'buffer_usage': self.get_buffer_usage(),
            'schema_type': self.schema_type.value,
            'assets': {}
        }
        
        for listing_id in self.listing_id_to_asset_no.keys():
            record = self.get_record(listing_id)
            if len(record.arr) > 0:
                asset_stats = {
                    'record_count': len(record.arr),
                    'timestamp_range': (int(np.min(record.arr['timestamp'])), int(np.max(record.arr['timestamp']))),
                    'price_range': (float(np.min(record.arr['price'])), float(np.max(record.arr['price']))),
                }
                summary['assets'][listing_id] = asset_stats
        
        return summary

    def __len__(self) -> int:
        """Return total number of records across all assets."""
        return self.get_total_record_count()

    def __contains__(self, listing_id: int) -> bool:
        """Check if a listing_id is tracked by this recorder."""
        return listing_id in self.listing_id_to_asset_no

    def __repr__(self) -> str:
        """String representation of the recorder."""
        return (f"Recorder(schema_type={self.schema_type.value}, "
                f"assets={len(self.listing_id_to_asset_no)}, "
                f"total_records={self.get_total_record_count()}, "
                f"buffer_size={len(self.records)})")

    def keys(self):
        """Return listing IDs tracked by this recorder."""
        return self.listing_id_to_asset_no.keys()

    def values(self):
        """Return Record objects for all assets."""
        return self.get_all_records().values()

    def items(self):
        """Return (listing_id, Record) pairs for all assets."""
        return self.get_all_records().items()

    def to_npz(self, file: str):
        """Persist all per-asset records to a compressed `.npz` file.

        Parameters
        ----------
        file : str
            Destination file path.
        """
        # Only save non-empty records for each asset
        kwargs = {}
        for asset_no in range(self.records.shape[1]):
            # Get the actual listing_id for this asset
            listing_id = next(k for k, v in self.listing_id_to_asset_no.items() if v == asset_no)
            # Only save records that have been written to (non-zero timestamps)
            valid_records = self.records[:self.at_i[asset_no], asset_no]
            if len(valid_records) > 0:
                kwargs[f"asset_{listing_id}"] = valid_records

        if kwargs:
            np.savez_compressed(file, **kwargs)
        else:
            # Create empty file if no records
            np.savez_compressed(file, empty=np.array([]))

    @classmethod
    def from_npz(cls, file: str, schema_type: SchemaType) -> 'Recorder':
        """Load a recorder from a saved .npz file.
        
        Parameters
        ----------
        file : str
            Path to the .npz file.
        schema_type : SchemaType
            The schema type used for the original recording.
            
        Returns
        -------
        Recorder
            A new Recorder instance loaded from the file.
            
        Note
        ----
        This is a simplified implementation. In practice, you might want to
        store metadata about the original recorder configuration.
        """
        data = np.load(file)
        
        # Extract listing IDs from the saved data
        listing_ids = []
        for key in data.keys():
            if key.startswith('asset_'):
                listing_id = int(key.split('_')[1])
                listing_ids.append(listing_id)
        
        if not listing_ids:
            raise ValueError("No asset data found in the file")
        
        # Create new recorder
        recorder = cls(listing_ids, schema_type)  # Default size
        
        # Load the data
        for listing_id in listing_ids:
            asset_data = data[f'asset_{listing_id}']
            if len(asset_data) > 0:
                asset_no = recorder.listing_id_to_asset_no[listing_id]
                # Ensure we have enough space
                while len(recorder.records) < len(asset_data):
                    recorder._resize_buffer()
                
                # Copy the data
                recorder.records[:len(asset_data), asset_no] = asset_data
                recorder.at_i[asset_no] = len(asset_data)
        
        return recorder
