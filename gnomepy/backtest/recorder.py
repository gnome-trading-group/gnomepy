from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any

import numpy as np

from gnomepy import SchemaBase
from gnomepy.data.types import (
    MBP10, MBP1, MBO, BBO, Trades, OHLCV,
    FIXED_PRICE_SCALE, FIXED_SIZE_SCALE, SchemaType
)

_BUILTIN_RECORDER_NAMES = frozenset({'market', 'intent'})

class RecordType(IntEnum):
    """Enumeration of supported record/event types."""
    MARKET = 1
    EXECUTION = 2
    INTENT = 3


class BaseRecorder(ABC):
    """Base class for recorders that handle a single record type.
    
    This abstract base class provides common functionality for recording
    structured data across multiple assets. Subclasses must implement
    the record-specific methods.
    
    Parameters
    ----------
    listing_ids : list[int]
        Unique listing identifiers for tracked assets.
    size : int, default 100_000
        Maximum number of events to record per asset.
    auto_resize : bool, default True
        Whether to automatically resize the buffer when it's full.
    """
    
    def __init__(self, listing_ids: list[int], size: int = 100_000, auto_resize: bool = True):
        num_assets = len(listing_ids)
        self.records = np.zeros((size, num_assets), self.get_dtype())
        self.at_i = np.zeros((num_assets,), dtype=np.int64)
        self.listing_id_to_asset_no = {v: k for k, v in enumerate(listing_ids)}
        self.auto_resize = auto_resize
        self.initial_size = size
    
    @abstractmethod
    def get_record_class(self):
        """Return the BaseRecord subclass to use for this recorder's data."""
        raise NotImplementedError
    
    def get_dtype(self) -> np.dtype:
        """Return the numpy dtype for this recorder's records."""
        return self.get_record_class().get_dtype()
    
    def _resize_buffer(self):
        """Dynamically resize the buffer when it's full."""
        old_size = len(self.records)
        new_size = old_size * 2  # Double the size
        
        # Create new larger buffer
        new_records = np.zeros((new_size, self.records.shape[1]), self.get_dtype())
        
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
        """Return non-empty rows for a given `listing_id` as a Record object."""
        if listing_id not in self.listing_id_to_asset_no:
            raise KeyError(f"Listing ID {listing_id} not found in recorder")

        asset_no = self.listing_id_to_asset_no[listing_id]
        count = self.at_i[asset_no]
        
        if count == 0:
            # Return empty array wrapped in record class
            empty_arr = np.array([], dtype=self.get_dtype())
            return self.get_record_class()(empty_arr)
        
        return self.get_record_class()(self.records[:count, asset_no].copy())
    
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


class MarketRecorder(BaseRecorder):
    """Recorder for market and execution events."""
    
    def __init__(self, listing_ids: list[int], schema_type: SchemaType, size: int = 100_000, auto_resize: bool = True):
        super().__init__(listing_ids, size, auto_resize)
        self.schema_type = schema_type
    
    def get_record_class(self):
        from gnomepy.backtest.stats.stats import MarketRecord
        return MarketRecord
    
    def log(
        self,
        event: RecordType,
        listing_id: int,
        timestamp: int,
        price: float = None,
        quantity: float = None,
        fee: float = None,
        fill_price: float = None,
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
        fill_price : float, optional
            Fill price at execution time.

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

        asset_no = self.listing_id_to_asset_no[listing_id]
        i = self.at_i[asset_no]

        if i >= len(self.records):
            if self.auto_resize:
                self._resize_buffer()
            else:
                raise IndexError(f"Buffer full for asset {listing_id}. Consider increasing size or enabling auto_resize.")

        self.records[i, asset_no]['event'] = event.value
        self.records[i, asset_no]['timestamp'] = timestamp
        self.records[i, asset_no]['price'] = price if price is not None else 0.0
        self.records[i, asset_no]['quantity'] = quantity if quantity is not None else 0.0
        self.records[i, asset_no]['fee'] = fee if fee is not None else 0.0
        self.records[i, asset_no]['fill_price'] = fill_price if fill_price is not None else 0.0

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
        self.records[i, asset_no]['price'] = price if price is not None else 0.0
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


class IntentRecorder(BaseRecorder):
    """Recorder for trading intent events."""
    
    def get_record_class(self):
        from gnomepy.backtest.stats.stats import IntentRecord
        return IntentRecord
    
    def log(
        self,
        listing_id: int,
        timestamp: int,
        side: str,
        confidence: float,
        price: float = None,
        flatten: bool = False
    ):
        """Append an intent record to the buffer.

        Parameters
        ----------
        listing_id : int
            Asset listing identifier.
        timestamp : int
            Event timestamp (ns or ms consistent with data source).
        side : str
            Order side ('B' for buy, 'S' or 'A' for sell/ask).
        confidence : float
            Confidence level (0.0 to 1.0).
        price : float, optional
            Limit order price, if applicable.
        flatten : bool, default False
            Whether this intent is to flatten the position.

        Raises
        ------
        KeyError
            If listing_id is not found in the recorder.
        IndexError
            If the per-asset buffer is full and auto_resize is disabled.
        ValueError
            If inputs are invalid.
        """
        # Input validation
        if listing_id not in self.listing_id_to_asset_no:
            raise KeyError(f"Listing ID {listing_id} not found in recorder. Available: {list(self.listing_id_to_asset_no.keys())}")
        
        if timestamp < 0:
            raise ValueError(f"Timestamp must be non-negative, got {timestamp}")
        
        if price is not None and price < 0:
            raise ValueError(f"Price must be non-negative, got {price}")
        
        if confidence < 0 or confidence > 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")
        
        if side not in ['B', 'S', 'A']:
            raise ValueError(f"Side must be 'B', 'S', or 'A', got {side}")
        
        asset_no = self.listing_id_to_asset_no[listing_id]
        i = self.at_i[asset_no]
        
        if i >= len(self.records):
            if self.auto_resize:
                self._resize_buffer()
            else:
                raise IndexError(f"Intent buffer full for asset {listing_id}. Consider increasing size or enabling auto_resize.")
        
        self.records[i, asset_no]['timestamp'] = timestamp
        self.records[i, asset_no]['side'] = side
        self.records[i, asset_no]['confidence'] = confidence
        self.records[i, asset_no]['price'] = price if price is not None else 0.0
        self.records[i, asset_no]['flatten'] = 1 if flatten else 0
        
        self.at_i[asset_no] += 1


def _make_generic_record_class(dtype: np.dtype):
    """Dynamically create a BaseRecord subclass for the given dtype."""
    from gnomepy.backtest.stats.stats import BaseRecord
    import pandas as pd

    _dtype = dtype

    class _GenericRecord(BaseRecord):
        @classmethod
        def get_dtype(cls) -> np.dtype:
            return _dtype

        def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].ffill()
            return df

    _GenericRecord.__name__ = 'GenericRecord'
    _GenericRecord.__qualname__ = 'GenericRecord'
    return _GenericRecord


class GenericRecorder(BaseRecorder):
    """Recorder for arbitrary model-specific values with a user-supplied numpy dtype.

    Parameters
    ----------
    listing_ids : list[int]
        Unique listing identifiers for tracked assets.
    dtype : np.dtype
        Numpy structured dtype describing the fields to record. Must include a
        field named ``timestamp`` of type ``i8``.
    record_class : type | None, default None
        A ``BaseRecord`` subclass to use when wrapping raw arrays. If *None* a
        class is generated dynamically from *dtype*.
    size : int, default 100_000
        Initial buffer capacity per asset.
    auto_resize : bool, default True
        Whether to grow the buffer automatically when it is full.
    """

    def __init__(
        self,
        listing_ids: list[int],
        dtype: np.dtype,
        record_class=None,
        size: int = 100_000,
        auto_resize: bool = True,
    ):
        self._dtype = np.dtype(dtype)
        self._record_class = record_class if record_class is not None else _make_generic_record_class(self._dtype)
        super().__init__(listing_ids, size, auto_resize)

    def get_record_class(self):
        return self._record_class

    def get_dtype(self) -> np.dtype:
        return self._dtype

    def log(self, listing_id: int, **fields):
        """Write a row of field values into the buffer for the given listing.

        Parameters
        ----------
        listing_id : int
            Asset listing identifier.
        **fields
            Keyword arguments matching field names in the recorder's dtype.
            A ``timestamp`` field is required.

        Raises
        ------
        KeyError
            If *listing_id* is not tracked by this recorder.
        ValueError
            If the ``timestamp`` field is missing or negative.
        IndexError
            If the buffer is full and *auto_resize* is disabled.
        """
        if listing_id not in self.listing_id_to_asset_no:
            raise KeyError(
                f"Listing ID {listing_id} not found in recorder. "
                f"Available: {list(self.listing_id_to_asset_no.keys())}"
            )

        timestamp = fields.get('timestamp')
        if timestamp is None:
            raise ValueError("A 'timestamp' field is required when calling log()")
        if timestamp < 0:
            raise ValueError(f"Timestamp must be non-negative, got {timestamp}")

        asset_no = self.listing_id_to_asset_no[listing_id]
        i = self.at_i[asset_no]

        if i >= len(self.records):
            if self.auto_resize:
                self._resize_buffer()
            else:
                raise IndexError(
                    f"Buffer full for asset {listing_id}. "
                    "Consider increasing size or enabling auto_resize."
                )

        for field_name in self._dtype.names:
            if field_name in fields:
                self.records[i, asset_no][field_name] = fields[field_name]

        self.at_i[asset_no] += 1


ModelValueRecorder = GenericRecorder


class Recorder:
    """Composite recorder that combines MarketRecorder, IntentRecorder, and ModelValueRecorder.
    
    This class provides backward compatibility by delegating to the individual recorders
    while maintaining the same interface as the original Recorder class.
    
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
        self.market_recorder = MarketRecorder(listing_ids, schema_type, size, auto_resize)
        self.intent_recorder = IntentRecorder(listing_ids, size, auto_resize)
        self.custom_recorders: dict[str, BaseRecorder] = {}

        # Expose common attributes for backward compatibility
        self.listing_id_to_asset_no = self.market_recorder.listing_id_to_asset_no
        self.schema_type = schema_type
        self.auto_resize = auto_resize
        self.initial_size = size
    
    # Delegate market recorder methods
    def log(self, *args, **kwargs):
        return self.market_recorder.log(*args, **kwargs)
    
    def log_market_event(self, *args, **kwargs):
        return self.market_recorder.log_market_event(*args, **kwargs)
    
    def get_record(self, listing_id: int):
        return self.market_recorder.get_record(listing_id)
    
    # Delegate intent recorder methods
    def log_intent(self, *args, **kwargs):
        return self.intent_recorder.log_intent(*args, **kwargs)
    
    def get_intent_record(self, listing_id: int):
        """Get intent records for a specific listing as an IntentRecord object.
        
        Parameters
        ----------
        listing_id : int
            Asset listing identifier.
            
        Returns
        -------
        IntentRecord
            IntentRecord object for the listing.
        """
        return self.intent_recorder.get_record(listing_id)
    
    # Buffer usage methods
    def get_buffer_usage(self) -> dict[int, float]:
        """Get market buffer usage statistics for each asset."""
        return self.market_recorder.get_buffer_usage()
    
    def get_intent_buffer_usage(self) -> dict[int, float]:
        """Get intent buffer usage statistics for each asset."""
        return self.intent_recorder.get_buffer_usage()

    # Custom recorder registry

    def register(self, name: str, recorder: BaseRecorder) -> None:
        """Register a custom recorder under *name*.

        Parameters
        ----------
        name : str
            Unique name for this recorder. Must not collide with built-in names
            (``'market'`` or ``'intent'``).
        recorder : BaseRecorder
            The recorder instance to register.

        Raises
        ------
        ValueError
            If *name* collides with a built-in recorder name or is already registered.
        """
        if name in _BUILTIN_RECORDER_NAMES:
            raise ValueError(
                f"'{name}' is a built-in recorder name. "
                f"Choose a name other than {sorted(_BUILTIN_RECORDER_NAMES)}."
            )
        if name in self.custom_recorders:
            raise ValueError(f"A custom recorder named '{name}' is already registered.")
        self.custom_recorders[name] = recorder

    def get_custom_recorder(self, name: str) -> BaseRecorder:
        """Return the custom recorder registered under *name*.

        Raises
        ------
        KeyError
            If no recorder with that name is registered.
        """
        if name not in self.custom_recorders:
            raise KeyError(f"No custom recorder named '{name}' is registered.")
        return self.custom_recorders[name]

    def get_custom_record(self, name: str, listing_id: int):
        """Shortcut for ``get_custom_recorder(name).get_record(listing_id)``."""
        return self.get_custom_recorder(name).get_record(listing_id)

    def get_all_custom_recorders(self) -> dict[str, BaseRecorder]:
        """Return a shallow copy of the custom recorders dict."""
        return dict(self.custom_recorders)

    def clear(self):
        """Clear all recorded data and reset counters."""
        self.market_recorder.clear()
        self.intent_recorder.clear()
        for recorder in self.custom_recorders.values():
            recorder.clear()
    
    def get_total_record_count(self) -> int:
        """Get total number of records across all assets and record types."""
        market_count = int(sum(self.market_recorder.at_i))
        intent_count = int(sum(self.intent_recorder.at_i))
        custom_count = sum(int(sum(r.at_i)) for r in self.custom_recorders.values())
        return market_count + intent_count + custom_count

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics for all recorders."""
        summary = {
            'total_records': self.get_total_record_count(),
            'market_buffer_usage': self.get_buffer_usage(),
            'intent_buffer_usage': self.get_intent_buffer_usage(),
            'schema_type': self.schema_type.value,
            'custom_recorders': list(self.custom_recorders.keys()),
            'assets': {},
        }

        for listing_id in self.listing_id_to_asset_no.keys():
            market_record = self.market_recorder.get_record(listing_id)
            intent_record = self.get_intent_record(listing_id)

            asset_stats = {}
            if len(market_record.arr) > 0:
                asset_stats['market_record_count'] = len(market_record.arr)
                asset_stats['market_timestamp_range'] = (int(np.min(market_record.arr['timestamp'])), int(np.max(market_record.arr['timestamp'])))
                asset_stats['market_price_range'] = (float(np.min(market_record.arr['price'])), float(np.max(market_record.arr['price'])))

            if len(intent_record.arr) > 0:
                asset_stats['intent_record_count'] = len(intent_record.arr)
                asset_stats['intent_timestamp_range'] = (int(np.min(intent_record.arr['timestamp'])), int(np.max(intent_record.arr['timestamp'])))

            for cname, crec in self.custom_recorders.items():
                record = crec.get_record(listing_id)
                if len(record.arr) > 0:
                    asset_stats[f'{cname}_record_count'] = len(record.arr)

            if asset_stats:
                summary['assets'][listing_id] = asset_stats

        return summary
    
    # Additional methods for backward compatibility
    def get_all_records(self):
        """Get records for all assets efficiently."""
        return self.market_recorder.get_all_records()
    
    def get_record_count(self, listing_id: int) -> int:
        """Get the number of market records for a specific asset."""
        if listing_id not in self.listing_id_to_asset_no:
            raise KeyError(f"Listing ID {listing_id} not found in recorder")
        asset_no = self.listing_id_to_asset_no[listing_id]
        return int(self.market_recorder.at_i[asset_no])
    
    def validate_data_integrity(self) -> dict[str, Any]:
        """Validate data integrity and consistency across all records."""
        issues = []
        stats = {
            'total_records': self.get_total_record_count(),
            'buffer_usage': self.get_buffer_usage(),
            'schema_type': self.schema_type.value,
            'issues': issues
        }
        
        # Check for timestamp ordering issues
        for listing_id in self.listing_id_to_asset_no.keys():
            record = self.market_recorder.get_record(listing_id)
            if len(record.arr) > 1:
                timestamps = record.arr['timestamp']
                if not np.all(timestamps[:-1] <= timestamps[1:]):
                    issues.append(f"Non-monotonic timestamps detected for asset {listing_id}")
        
        # Check for negative prices
        for listing_id in self.listing_id_to_asset_no.keys():
            record = self.market_recorder.get_record(listing_id)
            if len(record.arr) > 0:
                prices = record.arr['price']
                negative_prices = np.sum(prices < 0)
                if negative_prices > 0:
                    issues.append(f"Found {negative_prices} negative prices for asset {listing_id}")
        
        stats['is_valid'] = len(issues) == 0
        return stats
    
    def get_summary_stats(self) -> dict[str, Any]:
        """Get statistics for all recorded data (alias for get_summary for backward compatibility)."""
        return self.get_summary()
    
    def __len__(self) -> int:
        """Return total number of records across all assets."""
        return self.get_total_record_count()
    
    def __contains__(self, listing_id: int) -> bool:
        """Check if a listing_id is tracked by this recorder."""
        return listing_id in self.listing_id_to_asset_no
    
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

        Market records are stored under keys ``asset_{listing_id}``.
        Custom recorder data is stored under keys
        ``custom_{name}_asset_{listing_id}``.
        A metadata array listing custom recorder names and their dtype
        descriptions is saved under the key ``_custom_meta``.
        """
        kwargs = {}

        for asset_no in range(self.market_recorder.records.shape[1]):
            listing_id = next(k for k, v in self.listing_id_to_asset_no.items() if v == asset_no)
            valid_records = self.market_recorder.records[:self.market_recorder.at_i[asset_no], asset_no]
            if len(valid_records) > 0:
                kwargs[f"asset_{listing_id}"] = valid_records

        # Persist custom recorders
        custom_meta = []
        for cname, crec in self.custom_recorders.items():
            dtype_str = crec.get_dtype().str if hasattr(crec.get_dtype(), 'str') else str(crec.get_dtype())
            # For structured dtypes use descr
            dtype_descr = str(crec.get_dtype().descr)
            custom_meta.append(f"{cname}|||{dtype_descr}")
            for asset_no in range(crec.records.shape[1]):
                listing_id = next(k for k, v in crec.listing_id_to_asset_no.items() if v == asset_no)
                valid_records = crec.records[:crec.at_i[asset_no], asset_no]
                if len(valid_records) > 0:
                    kwargs[f"custom_{cname}_asset_{listing_id}"] = valid_records

        if custom_meta:
            kwargs['_custom_meta'] = np.array(custom_meta, dtype=object)

        # Always persist the listing IDs so from_npz can reconstruct the recorder
        # even when no market records were logged.
        kwargs['_listing_ids'] = np.array(
            list(self.listing_id_to_asset_no.keys()), dtype=np.int64
        )

        np.savez_compressed(file, **kwargs)

    @classmethod
    def from_npz(cls, file: str, schema_type: SchemaType) -> 'Recorder':
        """Load a recorder from a saved .npz file."""
        data = np.load(file, allow_pickle=True)

        # Prefer the explicitly-saved listing IDs; fall back to inferring from keys.
        if '_listing_ids' in data:
            listing_ids = list(map(int, data['_listing_ids']))
        else:
            listing_ids = []
            for key in data.keys():
                if key.startswith('asset_'):
                    listing_ids.append(int(key.split('_')[1]))

        if not listing_ids:
            raise ValueError("No asset data found in the file")

        recorder = cls(listing_ids, schema_type)

        for listing_id in listing_ids:
            key = f'asset_{listing_id}'
            if key not in data:
                continue
            asset_data = data[key]
            if len(asset_data) > 0:
                asset_no = recorder.listing_id_to_asset_no[listing_id]
                while len(recorder.market_recorder.records) < len(asset_data):
                    recorder.market_recorder._resize_buffer()

                recorder.market_recorder.records[:len(asset_data), asset_no] = asset_data
                recorder.market_recorder.at_i[asset_no] = len(asset_data)

        # Restore custom recorders
        if '_custom_meta' in data:
            import ast
            for meta_entry in data['_custom_meta']:
                cname, dtype_descr_str = str(meta_entry).split('|||', 1)
                dtype_descr = ast.literal_eval(dtype_descr_str)
                dtype = np.dtype(dtype_descr)

                crec = GenericRecorder(listing_ids, dtype)
                for listing_id in listing_ids:
                    key = f"custom_{cname}_asset_{listing_id}"
                    if key in data:
                        asset_data = data[key]
                        if len(asset_data) > 0:
                            asset_no = crec.listing_id_to_asset_no[listing_id]
                            while len(crec.records) < len(asset_data):
                                crec._resize_buffer()
                            crec.records[:len(asset_data), asset_no] = asset_data
                            crec.at_i[asset_no] = len(asset_data)
                recorder.custom_recorders[cname] = crec

        return recorder
    
    def __repr__(self):
        return (
            f"Recorder(schema_type={self.schema_type.value}, "
            f"assets={len(self.listing_id_to_asset_no)}, "
            f"total_records={self.get_total_record_count()}, "
            f"custom_recorders={len(self.custom_recorders)})"
        )
