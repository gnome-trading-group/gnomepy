"""
Comprehensive tests for SchemaBase encoding functionality.

Tests cover:
- All SchemaBase subclasses (MBO, MBP10, MBP1, BBO1S, BBO1M, Trades, OHLCV1S, OHLCV1M, OHLCV1H)
- Round-trip encoding/decoding (encode SchemaBase -> decode -> verify)
- Field mapping (snake_case to camelCase)
- Level flattening for MBP and BBO
- Null value handling
- Edge cases
"""

import pytest
import importlib_resources

from gnomepy.data.sbe import Schema
from gnomepy.data.types import (
    MBO, MBP10, MBP1, BBO1S, BBO1M, Trades, OHLCV1S, OHLCV1M, OHLCV1H,
    BidAskPair
)


@pytest.fixture(scope="module")
def schema():
    """Load the SBE schema once for all tests."""
    schema_file_module = "gnomepy.data.sbe"
    schema_file_name = "schema.xml"
    with importlib_resources.open_text(schema_file_module, schema_file_name) as f:
        return Schema.parse(f)


class TestMBOEncoding:
    """Test MBO encoding and round-trip."""

    def test_mbo_encode_basic(self, schema):
        """Test basic MBO encoding."""
        mbo = MBO(
            exchange_id=1,
            security_id=12345,
            timestamp_event=1234567890,
            timestamp_sent=1234567891,
            timestamp_recv=1234567892,
            order_id=123456789,
            price=50000,
            size=100,
            action='Add',
            side='Bid',
            flags=['topOfBook'],
            sequence=42,
        )

        encoded = mbo.encode()
        decoded = schema.decode(encoded)

        assert decoded.message_name == 'MBO'
        assert decoded.value['exchangeId'] == 1
        assert decoded.value['securityId'] == 12345
        assert decoded.value['timestampEvent'] == 1234567890
        assert decoded.value['timestampSent'] == 1234567891
        assert decoded.value['timestampRecv'] == 1234567892
        assert decoded.value['orderId'] == 123456789
        assert decoded.value['price'] == 50000
        assert decoded.value['size'] == 100
        assert decoded.value['action'] == 'Add'
        assert decoded.value['side'] == 'Bid'
        assert 'topOfBook' in decoded.value['flags']
        assert decoded.value['sequence'] == 42

    def test_mbo_encode_with_null_values(self, schema):
        """Test MBO encoding with null values."""
        mbo = MBO(
            exchange_id=1,
            security_id=1,
            timestamp_event=0,
            timestamp_sent=None,
            timestamp_recv=0,
            order_id=999,
            price=100,
            size=10,
            action='Add',
            side='Bid',
            flags=[],
            sequence=None,
        )

        encoded = mbo.encode()
        decoded = schema.decode(encoded)

        assert decoded.value['timestampSent'] is None
        assert decoded.value['sequence'] is None
        assert decoded.value['flags'] == []

    def test_mbo_encode_all_actions(self, schema):
        """Test MBO encoding with all action types."""
        for action in ['Add', 'Modify', 'Cancel', 'Clear']:
            mbo = MBO(
                exchange_id=1,
                security_id=1,
                timestamp_event=0,
                timestamp_sent=0,
                timestamp_recv=0,
                order_id=999,
                price=100,
                size=10,
                action=action,
                side='Bid',
                flags=[],
                sequence=0,
            )

            encoded = mbo.encode()
            decoded = schema.decode(encoded)

            assert decoded.value['action'] == action


class TestMBP10Encoding:
    """Test MBP10 encoding and round-trip."""

    def test_mbp10_encode_basic(self, schema):
        """Test basic MBP10 encoding with levels."""
        levels = [
            BidAskPair(bid_px=100000, ask_px=100100, bid_sz=50, ask_sz=60, bid_ct=5, ask_ct=6)
            for _ in range(10)
        ]

        mbp10 = MBP10(
            exchange_id=1,
            security_id=12345,
            timestamp_event=1234567890,
            timestamp_sent=1234567891,
            timestamp_recv=1234567892,
            price=100000,
            size=50,
            action='Add',
            side='Bid',
            flags=['marketByPrice'],
            sequence=42,
            depth=10,
            levels=levels,
        )

        encoded = mbp10.encode()
        decoded = schema.decode(encoded)

        assert decoded.message_name == 'MBP10'
        assert decoded.value['exchangeId'] == 1
        assert decoded.value['securityId'] == 12345
        assert decoded.value['depth'] == 10
        
        # Check that levels were flattened correctly
        for i in range(10):
            assert decoded.value[f'bidPrice{i}'] == 100000
            assert decoded.value[f'askPrice{i}'] == 100100
            assert decoded.value[f'bidSize{i}'] == 50
            assert decoded.value[f'askSize{i}'] == 60
            assert decoded.value[f'bidCount{i}'] == 5
            assert decoded.value[f'askCount{i}'] == 6

    def test_mbp10_encode_partial_levels(self, schema):
        """Test MBP10 encoding with fewer than 10 levels."""
        levels = [
            BidAskPair(bid_px=100000 + i*100, ask_px=100100 + i*100, 
                      bid_sz=50-i, ask_sz=60-i, bid_ct=5, ask_ct=6)
            for i in range(3)
        ]

        mbp10 = MBP10(
            exchange_id=1,
            security_id=1,
            timestamp_event=0,
            timestamp_sent=0,
            timestamp_recv=0,
            price=100000,
            size=50,
            action='Add',
            side='Bid',
            flags=[],
            sequence=0,
            depth=3,
            levels=levels,
        )

        encoded = mbp10.encode()
        decoded = schema.decode(encoded)

        # Check first 3 levels have data
        for i in range(3):
            assert decoded.value[f'bidPrice{i}'] == 100000 + i*100
            assert decoded.value[f'askPrice{i}'] == 100100 + i*100
        
        # Check remaining levels are None
        for i in range(3, 10):
            assert decoded.value[f'bidPrice{i}'] is None
            assert decoded.value[f'askPrice{i}'] is None


class TestMBP1Encoding:
    """Test MBP1 encoding and round-trip."""

    def test_mbp1_encode_basic(self, schema):
        """Test basic MBP1 encoding."""
        level = BidAskPair(bid_px=100000, ask_px=100100, bid_sz=50, ask_sz=60, bid_ct=5, ask_ct=6)

        mbp1 = MBP1(
            exchange_id=1,
            security_id=12345,
            timestamp_event=1234567890,
            timestamp_sent=1234567891,
            timestamp_recv=1234567892,
            price=100000,
            size=50,
            action='Add',
            side='Bid',
            flags=[],
            sequence=42,
            depth=1,
            levels=[level],
        )

        encoded = mbp1.encode()
        decoded = schema.decode(encoded)

        assert decoded.message_name == 'MBP1'
        assert decoded.value['bidPrice0'] == 100000
        assert decoded.value['askPrice0'] == 100100
        assert decoded.value['bidSize0'] == 50
        assert decoded.value['askSize0'] == 60


class TestBBOEncoding:
    """Test BBO encoding and round-trip."""

    def test_bbo1s_encode_basic(self, schema):
        """Test BBO1S encoding."""
        level = BidAskPair(bid_px=100000, ask_px=100100, bid_sz=50, ask_sz=60, bid_ct=5, ask_ct=6)

        bbo = BBO1S(
            exchange_id=1,
            security_id=12345,
            timestamp_event=1234567890,
            timestamp_recv=1234567892,
            price=100000,
            size=50,
            side='Bid',
            flags=['topOfBook'],
            sequence=42,
            levels=[level],
        )

        encoded = bbo.encode()
        decoded = schema.decode(encoded)

        assert decoded.message_name == 'BBO1S'
        assert decoded.value['exchangeId'] == 1
        assert decoded.value['bidPrice0'] == 100000
        assert decoded.value['askPrice0'] == 100100

    def test_bbo1m_encode_basic(self, schema):
        """Test BBO1M encoding."""
        level = BidAskPair(bid_px=100000, ask_px=100100, bid_sz=50, ask_sz=60, bid_ct=5, ask_ct=6)

        bbo = BBO1M(
            exchange_id=1,
            security_id=12345,
            timestamp_event=1234567890,
            timestamp_recv=1234567892,
            price=100000,
            size=50,
            side='Bid',
            flags=[],
            sequence=42,
            levels=[level],
        )

        encoded = bbo.encode()
        decoded = schema.decode(encoded)

        assert decoded.message_name == 'BBO1M'
        assert decoded.value['exchangeId'] == 1


class TestTradesEncoding:
    """Test Trades encoding and round-trip."""

    def test_trades_encode_basic(self, schema):
        """Test basic Trades encoding."""
        trades = Trades(
            exchange_id=1,
            security_id=12345,
            timestamp_event=1234567890,
            timestamp_sent=1234567891,
            timestamp_recv=1234567892,
            price=50000,
            size=100,
            action='Trade',
            side='Bid',
            flags=[],
            sequence=42,
            depth=1,
        )

        encoded = trades.encode()
        decoded = schema.decode(encoded)

        assert decoded.message_name == 'Trades'
        assert decoded.value['exchangeId'] == 1
        assert decoded.value['securityId'] == 12345
        assert decoded.value['price'] == 50000
        assert decoded.value['size'] == 100
        assert decoded.value['action'] == 'Trade'


class TestOHLCVEncoding:
    """Test OHLCV encoding and round-trip."""

    def test_ohlcv1s_encode_basic(self, schema):
        """Test OHLCV1S encoding."""
        ohlcv = OHLCV1S(
            exchange_id=1,
            security_id=12345,
            timestamp_event=1234567890,
            open=100000,
            high=105000,
            low=99000,
            close=102000,
            volume=5000000,
        )

        encoded = ohlcv.encode()
        decoded = schema.decode(encoded)

        assert decoded.message_name == 'OHLCV1S'
        assert decoded.value['exchangeId'] == 1
        assert decoded.value['securityId'] == 12345
        assert decoded.value['timestampEvent'] == 1234567890
        assert decoded.value['open'] == 100000
        assert decoded.value['high'] == 105000
        assert decoded.value['low'] == 99000
        assert decoded.value['close'] == 102000
        assert decoded.value['volume'] == 5000000

    def test_ohlcv1m_encode_basic(self, schema):
        """Test OHLCV1M encoding."""
        ohlcv = OHLCV1M(
            exchange_id=1,
            security_id=12345,
            timestamp_event=1234567890,
            open=100000,
            high=105000,
            low=99000,
            close=102000,
            volume=5000000,
        )

        encoded = ohlcv.encode()
        decoded = schema.decode(encoded)

        assert decoded.message_name == 'OHLCV1M'

    def test_ohlcv1h_encode_basic(self, schema):
        """Test OHLCV1H encoding."""
        ohlcv = OHLCV1H(
            exchange_id=1,
            security_id=12345,
            timestamp_event=1234567890,
            open=100000,
            high=105000,
            low=99000,
            close=102000,
            volume=5000000,
        )

        encoded = ohlcv.encode()
        decoded = schema.decode(encoded)

        assert decoded.message_name == 'OHLCV1H'


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_multiple_flags(self, schema):
        """Test encoding with multiple flags."""
        mbo = MBO(
            exchange_id=1,
            security_id=1,
            timestamp_event=0,
            timestamp_sent=0,
            timestamp_recv=0,
            order_id=999,
            price=100,
            size=10,
            action='Add',
            side='Bid',
            flags=['lastMessage', 'topOfBook', 'snapshot', 'marketByPrice'],
            sequence=0,
        )

        encoded = mbo.encode()
        decoded = schema.decode(encoded)

        assert set(decoded.value['flags']) == {'lastMessage', 'topOfBook', 'snapshot', 'marketByPrice'}

    def test_max_uint64_values(self, schema):
        """Test encoding with maximum uint64 values."""
        mbo = MBO(
            exchange_id=1,
            security_id=1,
            timestamp_event=18446744073709551615,  # Max uint64
            timestamp_sent=18446744073709551615,
            timestamp_recv=18446744073709551615,
            order_id=18446744073709551615,  # Max uint64
            price=100,
            size=10,
            action='Add',
            side='Bid',
            flags=[],
            sequence=0,
        )

        encoded = mbo.encode()
        decoded = schema.decode(encoded)

        assert decoded.value['timestampEvent'] == 18446744073709551615
        assert decoded.value['orderId'] == 18446744073709551615

    def test_zero_values(self, schema):
        """Test encoding with zero values."""
        ohlcv = OHLCV1S(
            exchange_id=0,
            security_id=0,
            timestamp_event=0,
            open=0,
            high=0,
            low=0,
            close=0,
            volume=0,
        )

        encoded = ohlcv.encode()
        decoded = schema.decode(encoded)

        assert decoded.value['exchangeId'] == 0
        assert decoded.value['open'] == 0
        assert decoded.value['volume'] == 0

    def test_round_trip_preserves_data(self, schema):
        """Test that round-trip encoding/decoding preserves all data."""
        original_mbo = MBO(
            exchange_id=123,
            security_id=456789,
            timestamp_event=9876543210,
            timestamp_sent=9876543211,
            timestamp_recv=9876543212,
            order_id=987654321012345,
            price=123456789,
            size=987654,
            action='Modify',
            side='Ask',
            flags=['topOfBook', 'snapshot'],
            sequence=999,
        )

        # Encode
        encoded = original_mbo.encode()

        # Decode
        decoded = schema.decode(encoded)

        # Create new MBO from decoded message
        reconstructed_mbo = MBO.from_message(decoded)

        # Verify all fields match
        assert reconstructed_mbo.exchange_id == original_mbo.exchange_id
        assert reconstructed_mbo.security_id == original_mbo.security_id
        assert reconstructed_mbo.timestamp_event == original_mbo.timestamp_event
        assert reconstructed_mbo.timestamp_sent == original_mbo.timestamp_sent
        assert reconstructed_mbo.timestamp_recv == original_mbo.timestamp_recv
        assert reconstructed_mbo.order_id == original_mbo.order_id
        assert reconstructed_mbo.price == original_mbo.price
        assert reconstructed_mbo.size == original_mbo.size
        assert reconstructed_mbo.action == original_mbo.action
        assert reconstructed_mbo.side == original_mbo.side
        assert set(reconstructed_mbo.flags) == set(original_mbo.flags)
        assert reconstructed_mbo.sequence == original_mbo.sequence


class TestSchemaCache:
    """Test that schema caching works correctly."""

    def test_schema_is_cached(self):
        """Test that schema is loaded once and cached."""
        # Create first instance and encode
        mbo1 = MBO(
            exchange_id=1, security_id=1, timestamp_event=0,
            timestamp_sent=0, timestamp_recv=0, order_id=111,
            price=100, size=10, action='Add', side='Bid',
            flags=[], sequence=0,
        )
        mbo1.encode()

        # Get the cached schema
        cached_schema = MBO._schema

        # Create second instance and encode
        mbo2 = MBO(
            exchange_id=2, security_id=2, timestamp_event=0,
            timestamp_sent=0, timestamp_recv=0, order_id=222,
            price=200, size=20, action='Add', side='Ask',
            flags=[], sequence=0,
        )
        mbo2.encode()

        # Verify same schema instance is used
        assert MBO._schema is cached_schema
        assert MBO._schema is not None

