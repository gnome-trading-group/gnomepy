"""
Comprehensive tests for SBE encoding and decoding functionality.

Tests cover:
- All message types defined in schema.xml
- Primitive types (uint8, uint16, uint32, uint64, int64, char)
- Enums (Action, Side, OrderType, TimeInForce, ExecType, OrderStatus, RejectReason)
- Sets (MarketUpdateFlags, OrderFlags)
- Null values and optional fields
- Edge cases and boundary values
- Round-trip encoding/decoding
"""

import struct
import pytest
import importlib_resources

from gnomepy.data.sbe import Schema


@pytest.fixture(scope="module")
def schema():
    """Load the SBE schema once for all tests."""
    schema_file_module = "gnomepy.data.sbe"
    schema_file_name = "schema.xml"
    with importlib_resources.open_text(schema_file_module, schema_file_name) as f:
        return Schema.parse(f)


class TestPrimitiveTypes:
    """Test encoding and decoding of primitive types."""

    def test_uint_types(self, schema):
        """Test round-trip encoding/decoding of unsigned integer types."""
        message = schema.messages[1]  # MBO message

        obj = {
            'exchangeId': 1,
            'securityId': 12345,
            'timestampEvent': 1234567890,
            'timestampSent': 9876543210,
            'timestampRecv': 1111111111,
            'orderId': 999999999999,
            'price': 100000,
            'size': 500,
            'action': 'Add',
            'side': 'Bid',
            'flags': [],
            'sequence': 42,
        }

        encoded = schema.encode(message, obj)
        decoded = schema.decode(encoded)

        assert decoded.message_name == 'MBO'
        assert decoded.value['exchangeId'] == 1
        assert decoded.value['securityId'] == 12345
        assert decoded.value['timestampEvent'] == 1234567890
        assert decoded.value['orderId'] == 999999999999
        assert decoded.value['size'] == 500

    def test_int64_type(self, schema):
        """Test round-trip encoding/decoding of int64 type."""
        message = schema.messages[1]  # MBO message

        obj = {
            'exchangeId': 1,
            'securityId': 1,
            'timestampEvent': 0,
            'timestampSent': 0,
            'timestampRecv': 0,
            'orderId': 1,
            'price': -1000000,  # Negative price
            'size': 10,
            'action': 'Add',
            'side': 'Bid',
            'flags': [],
            'sequence': 0,
        }

        encoded = schema.encode(message, obj)
        decoded = schema.decode(encoded)

        assert decoded.value['price'] == -1000000

    def test_char_array(self, schema):
        """Test round-trip encoding/decoding of char array (clientOid field)."""
        message = schema.messages[10]  # Order message

        obj = {
            'exchangeId': 1,
            'securityId': 1,
            'timestampSend': 1234567890,
            'clientOid': 'ABC123XYZ',
            'price': 50000,
            'size': 100,
            'side': 'Bid',
            'flags': [],
            'orderType': 'LIMIT',
            'timeInForce': 'GOOD_TILL_CANCELED',
        }

        encoded = schema.encode(message, obj)
        decoded = schema.decode(encoded)

        assert decoded.message_name == 'Order'
        assert decoded.value['clientOid'] == 'ABC123XYZ'


class TestEnums:
    """Test encoding and decoding of enums."""

    def test_char_enum(self, schema):
        """Test round-trip encoding/decoding of character-based enums (Action, Side)."""
        message = schema.messages[1]  # MBO message

        # Test Action enum
        for action in ['Add', 'Modify', 'Cancel', 'Clear']:
            obj = {
                'exchangeId': 1,
                'securityId': 1,
                'timestampEvent': 0,
                'timestampSent': 0,
                'timestampRecv': 0,
                'orderId': 1,
                'price': 100,
                'size': 10,
                'action': action,
                'side': 'Bid',
                'flags': [],
                'sequence': 0,
            }
            encoded = schema.encode(message, obj)
            decoded = schema.decode(encoded)
            assert decoded.value['action'] == action

        # Test Side enum
        for side in ['Bid', 'Ask']:
            obj['side'] = side
            encoded = schema.encode(message, obj)
            decoded = schema.decode(encoded)
            assert decoded.value['side'] == side

    def test_uint8_enum(self, schema):
        """Test round-trip encoding/decoding of uint8-based enums (OrderType, TimeInForce, etc.)."""
        message = schema.messages[10]  # Order message

        # Test OrderType enum
        order_types = ['LIMIT', 'MARKET']
        for order_type in order_types:
            obj = {
                'exchangeId': 1,
                'securityId': 1,
                'timestampSend': 0,
                'clientOid': 'TEST',
                'price': 100,
                'size': 10,
                'side': 'Bid',
                'flags': [],
                'orderType': order_type,
                'timeInForce': 'GOOD_TILL_CANCELED',
            }
            encoded = schema.encode(message, obj)
            decoded = schema.decode(encoded)
            assert decoded.value['orderType'] == order_type

    def test_exec_type_enum(self, schema):
        """Test round-trip encoding/decoding of ExecType enum."""
        message = schema.messages[11]  # OrderExecutionReport message

        exec_types = ['NEW', 'CANCEL', 'FILL', 'PARTIAL_FILL', 'REJECT', 'CANCEL_REJECT', 'EXPIRE']
        for exec_type in exec_types:
            obj = {
                'exchangeId': 1,
                'securityId': 1,
                'clientOid': 'TEST',
                'execType': exec_type,
                'orderStatus': 'NEW',
                'filledQty': 0,
                'fillPrice': 0,
                'cumulativeQty': 0,
                'leavesQty': 100,
                'timestampEvent': 0,
                'timestampRecv': 0,
                'flags': [],
                'rejectReason': None,
            }
            encoded = schema.encode(message, obj)
            decoded = schema.decode(encoded)
            assert decoded.value['execType'] == exec_type


class TestSets:
    """Test encoding and decoding of sets (bitfields)."""

    def test_empty_set(self, schema):
        """Test round-trip encoding/decoding of empty set."""
        message = schema.messages[1]  # MBO message

        obj = {
            'exchangeId': 1,
            'securityId': 1,
            'timestampEvent': 0,
            'timestampSent': 0,
            'timestampRecv': 0,
            'orderId': 1,
            'price': 100,
            'size': 10,
            'action': 'Add',
            'side': 'Bid',
            'flags': [],
            'sequence': 0,
        }

        encoded = schema.encode(message, obj)
        decoded = schema.decode(encoded)

        assert decoded.value['flags'] == []

    def test_single_flag_set(self, schema):
        """Test round-trip encoding/decoding of set with single flag."""
        message = schema.messages[1]  # MBO message

        for flag in ['lastMessage', 'topOfBook', 'snapshot', 'marketByPrice']:
            obj = {
                'exchangeId': 1,
                'securityId': 1,
                'timestampEvent': 0,
                'timestampSent': 0,
                'timestampRecv': 0,
                'orderId': 1,
                'price': 100,
                'size': 10,
                'action': 'Add',
                'side': 'Bid',
                'flags': [flag],
                'sequence': 0,
            }

            encoded = schema.encode(message, obj)
            decoded = schema.decode(encoded)

            assert flag in decoded.value['flags']

    def test_multiple_flags_set(self, schema):
        """Test round-trip encoding/decoding of set with multiple flags."""
        message = schema.messages[1]  # MBO message

        obj = {
            'exchangeId': 1,
            'securityId': 1,
            'timestampEvent': 0,
            'timestampSent': 0,
            'timestampRecv': 0,
            'orderId': 1,
            'price': 100,
            'size': 10,
            'action': 'Add',
            'side': 'Bid',
            'flags': ['lastMessage', 'topOfBook', 'snapshot'],
            'sequence': 0,
        }

        encoded = schema.encode(message, obj)
        decoded = schema.decode(encoded)

        assert set(decoded.value['flags']) == {'lastMessage', 'topOfBook', 'snapshot'}


class TestNullValues:
    """Test encoding and decoding of null values."""

    def test_null_timestamp(self, schema):
        """Test round-trip encoding/decoding of null timestamp."""
        message = schema.messages[1]  # MBO message

        obj = {
            'exchangeId': 1,
            'securityId': 1,
            'timestampEvent': None,  # Null timestamp
            'timestampSent': 0,
            'timestampRecv': 0,
            'orderId': 1,
            'price': 100,
            'size': 10,
            'action': 'Add',
            'side': 'Bid',
            'flags': [],
            'sequence': 0,
        }

        encoded = schema.encode(message, obj)
        decoded = schema.decode(encoded)

        assert decoded.value['timestampEvent'] is None

    def test_null_price(self, schema):
        """Test round-trip encoding/decoding of null price."""
        message = schema.messages[1]  # MBO message

        obj = {
            'exchangeId': 1,
            'securityId': 1,
            'timestampEvent': 0,
            'timestampSent': 0,
            'timestampRecv': 0,
            'orderId': 1,
            'price': None,  # Null price
            'size': 10,
            'action': 'Add',
            'side': 'Bid',
            'flags': [],
            'sequence': 0,
        }

        encoded = schema.encode(message, obj)
        decoded = schema.decode(encoded)

        assert decoded.value['price'] is None


class TestMessageTypes:
    """Test encoding and decoding of all message types."""

    def test_mbo_message(self, schema):
        """Test round-trip encoding/decoding of MBO message."""
        message = schema.messages[1]

        obj = {
            'exchangeId': 1,
            'securityId': 12345,
            'timestampEvent': 1234567890,
            'timestampSent': 1234567891,
            'timestampRecv': 1234567892,
            'orderId': 999999,
            'price': 50000,
            'size': 100,
            'action': 'Add',
            'side': 'Bid',
            'flags': ['topOfBook', 'marketByPrice'],
            'sequence': 42,
        }

        encoded = schema.encode(message, obj)
        decoded = schema.decode(encoded)

        assert decoded.message_name == 'MBO'
        assert decoded.value['exchangeId'] == 1
        assert decoded.value['securityId'] == 12345
        assert decoded.value['orderId'] == 999999
        assert decoded.value['price'] == 50000
        assert decoded.value['action'] == 'Add'
        assert set(decoded.value['flags']) == {'topOfBook', 'marketByPrice'}

    def test_ohlcv_messages(self, schema):
        """Test round-trip encoding/decoding of OHLCV messages."""
        for msg_id in [7, 8, 9]:  # OHLCV1S, OHLCV1M, OHLCV1H
            message = schema.messages[msg_id]

            obj = {
                'exchangeId': 1,
                'securityId': 12345,
                'timestampEvent': 1234567890,
                'open': 100000,
                'high': 105000,
                'low': 99000,
                'close': 102000,
                'volume': 5000000,
            }

            encoded = schema.encode(message, obj)
            decoded = schema.decode(encoded)

            assert decoded.value['open'] == 100000
            assert decoded.value['high'] == 105000
            assert decoded.value['low'] == 99000
            assert decoded.value['close'] == 102000
            assert decoded.value['volume'] == 5000000

    def test_order_message(self, schema):
        """Test round-trip encoding/decoding of Order message."""
        message = schema.messages[10]

        obj = {
            'exchangeId': 1,
            'securityId': 12345,
            'timestampSend': 1609459200000000000,
            'clientOid': 'ORDER_ABC_123',
            'price': 50000,
            'size': 100,
            'side': 'Bid',
            'flags': [],
            'orderType': 'LIMIT',
            'timeInForce': 'GOOD_TILL_CANCELED',
        }

        encoded = schema.encode(message, obj)
        decoded = schema.decode(encoded)

        assert decoded.message_name == 'Order'
        assert decoded.value['exchangeId'] == 1
        assert decoded.value['securityId'] == 12345
        assert decoded.value['clientOid'] == 'ORDER_ABC_123'
        assert decoded.value['price'] == 50000
        assert decoded.value['side'] == 'Bid'
        assert decoded.value['orderType'] == 'LIMIT'

    def test_order_execution_report(self, schema):
        """Test round-trip encoding/decoding of OrderExecutionReport message."""
        message = schema.messages[11]

        obj = {
            'exchangeId': 1,
            'securityId': 12345,
            'clientOid': 'ORDER_XYZ_789',
            'execType': 'PARTIAL_FILL',
            'orderStatus': 'PARTIALLY_FILLED',
            'filledQty': 50,
            'fillPrice': 50100,
            'cumulativeQty': 50,
            'leavesQty': 50,
            'timestampEvent': 1609459200000000000,
            'timestampRecv': 1609459200000000100,
            'flags': [],
            'rejectReason': None,
        }

        encoded = schema.encode(message, obj)
        decoded = schema.decode(encoded)

        assert decoded.message_name == 'OrderExecutionReport'
        assert decoded.value['clientOid'] == 'ORDER_XYZ_789'
        assert decoded.value['execType'] == 'PARTIAL_FILL'
        assert decoded.value['orderStatus'] == 'PARTIALLY_FILLED'
        assert decoded.value['filledQty'] == 50

    def test_cancel_order_message(self, schema):
        """Test round-trip encoding/decoding of CancelOrder message."""
        message = schema.messages[12]

        obj = {
            'exchangeId': 1,
            'securityId': 12345,
            'clientOid': 'ORDER_TO_CANCEL',
        }

        encoded = schema.encode(message, obj)
        decoded = schema.decode(encoded)

        assert decoded.message_name == 'CancelOrder'
        assert decoded.value['exchangeId'] == 1
        assert decoded.value['securityId'] == 12345
        assert decoded.value['clientOid'] == 'ORDER_TO_CANCEL'


class TestHeaders:
    """Test encoding and decoding of message headers."""

    def test_header_contains_template_id(self, schema):
        """Test that encoded message header contains correct template ID."""
        message = schema.messages[1]  # MBO message

        obj = {
            'exchangeId': 1,
            'securityId': 1,
            'timestampEvent': 0,
            'timestampSent': 0,
            'timestampRecv': 0,
            'orderId': 1,
            'price': 100,
            'size': 10,
            'action': 'Add',
            'side': 'Bid',
            'flags': [],
            'sequence': 0,
        }

        encoded = schema.encode(message, obj)
        header = schema.decode_header(encoded)

        assert header['templateId'] == 1
        assert header['schemaId'] == 1
        assert header['version'] == 0

    def test_custom_header_values(self, schema):
        """Test that custom header values are preserved."""
        message = schema.messages[1]  # MBO message

        obj = {
            'exchangeId': 1,
            'securityId': 1,
            'timestampEvent': 0,
            'timestampSent': 0,
            'timestampRecv': 0,
            'orderId': 1,
            'price': 100,
            'size': 10,
            'action': 'Add',
            'side': 'Bid',
            'flags': [],
            'sequence': 0,
        }

        # Encode with custom blockLength
        custom_header = {'blockLength': 999}
        encoded = schema.encode(message, obj, header=custom_header)
        header = schema.decode_header(encoded)

        assert header['blockLength'] == 999

    def test_different_message_headers(self, schema):
        """Test that different message types have correct headers."""
        test_cases = [
            (1, 'MBO'),
            (7, 'OHLCV1S'),
            (10, 'Order'),
        ]

        for template_id, expected_name in test_cases:
            message = schema.messages[template_id]

            # Create minimal valid object for each message type
            if template_id == 1:
                obj = {
                    'exchangeId': 1, 'securityId': 1, 'timestampEvent': 0,
                    'timestampSent': 0, 'timestampRecv': 0, 'orderId': 1,
                    'price': 100, 'size': 10, 'action': 'Add', 'side': 'Bid',
                    'flags': [], 'sequence': 0,
                }
            elif template_id == 7:
                obj = {
                    'exchangeId': 1, 'securityId': 1, 'timestampEvent': 0,
                    'open': 100, 'high': 100, 'low': 100, 'close': 100, 'volume': 100,
                }
            elif template_id == 10:
                obj = {
                    'exchangeId': 1, 'securityId': 1, 'timestampSend': 0,
                    'clientOid': 'TEST', 'price': 100, 'size': 10, 'side': 'Bid',
                    'flags': [], 'orderType': 'LIMIT', 'timeInForce': 'GOOD_TILL_CANCELED',
                }

            encoded = schema.encode(message, obj)
            decoded = schema.decode(encoded)

            assert decoded.message_name == expected_name
            assert decoded.header['templateId'] == template_id


class TestEdgeCases:
    """Test edge cases and boundary values."""

    def test_max_uint64(self, schema):
        """Test round-trip encoding/decoding of maximum uint64 value."""
        message = schema.messages[1]  # MBO message

        obj = {
            'exchangeId': 1,
            'securityId': 1,
            'timestampEvent': 18446744073709551615,  # Max uint64
            'timestampSent': 0,
            'timestampRecv': 0,
            'orderId': 1,
            'price': 100,
            'size': 10,
            'action': 'Add',
            'side': 'Bid',
            'flags': [],
            'sequence': 0,
        }

        encoded = schema.encode(message, obj)
        decoded = schema.decode(encoded)

        assert decoded.value['timestampEvent'] == 18446744073709551615

    def test_zero_values(self, schema):
        """Test round-trip encoding/decoding of zero values."""
        message = schema.messages[1]  # MBO message

        obj = {
            'exchangeId': 0,
            'securityId': 0,
            'timestampEvent': 0,
            'timestampSent': 0,
            'timestampRecv': 0,
            'orderId': 0,
            'price': 0,
            'size': 0,
            'action': 'Add',
            'side': 'Bid',
            'flags': [],
            'sequence': 0,
        }

        encoded = schema.encode(message, obj)
        decoded = schema.decode(encoded)

        assert decoded.value['exchangeId'] == 0
        assert decoded.value['securityId'] == 0
        assert decoded.value['size'] == 0

    def test_empty_client_oid(self, schema):
        """Test round-trip encoding/decoding with empty client OID."""
        message = schema.messages[10]

        obj = {
            'exchangeId': 1,
            'securityId': 1,
            'timestampSend': 0,
            'clientOid': '',  # Empty string
            'price': 100,
            'size': 10,
            'side': 'Bid',
            'flags': [],
            'orderType': 'LIMIT',
            'timeInForce': 'GOOD_TILL_CANCELED',
        }

        encoded = schema.encode(message, obj)
        decoded = schema.decode(encoded)

        assert decoded.value['clientOid'] == ''

    def test_all_market_update_flags(self, schema):
        """Test round-trip encoding/decoding with all MarketUpdateFlags set."""
        message = schema.messages[1]  # MBO message

        all_flags = ['lastMessage', 'topOfBook', 'snapshot', 'marketByPrice',
                     'badTimestampRecv', 'maybeBadBook']

        obj = {
            'exchangeId': 1,
            'securityId': 1,
            'timestampEvent': 0,
            'timestampSent': 0,
            'timestampRecv': 0,
            'orderId': 1,
            'price': 100,
            'size': 10,
            'action': 'Add',
            'side': 'Bid',
            'flags': all_flags,
            'sequence': 0,
        }

        encoded = schema.encode(message, obj)
        decoded = schema.decode(encoded)

        assert set(decoded.value['flags']) == set(all_flags)

