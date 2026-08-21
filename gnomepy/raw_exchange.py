import struct
import zstandard

HEADER_FORMAT = "<qI"  # little-endian: int64 recv_timestamp_nanos, uint32 payload_length
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def read_raw_messages(data: bytes) -> list[tuple[int, bytes]]:
    """Parse a decompressed raw exchange capture file.

    Returns a list of (recv_timestamp_nanos, raw_payload_bytes) tuples.
    """
    messages = []
    offset = 0
    while offset + HEADER_SIZE <= len(data):
        recv_ts, payload_len = struct.unpack_from(HEADER_FORMAT, data, offset)
        offset += HEADER_SIZE
        payload = data[offset : offset + payload_len]
        offset += payload_len
        messages.append((recv_ts, payload))
    return messages


def decompress_and_read(compressed: bytes) -> list[tuple[int, bytes]]:
    """Decompress a zstd-compressed raw exchange capture file and parse its messages."""
    data = zstandard.ZstdDecompressor().decompress(compressed)
    return read_raw_messages(data)
