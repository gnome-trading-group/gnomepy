from __future__ import annotations


class StreamDirectory:
    """The streams a strategy reads and the listings they carry (wraps group.gnometrading.md.StreamDirectory).

    Market data names only its stream; orders, intents and execution reports name a listing by exchange and
    security. Key strategy state by listing id and resolve each side here.
    """

    def __init__(self, java_directory):
        self._java = java_directory
        self._none = int(java_directory.NONE)

    def listing_id(self, stream_id: int) -> int | None:
        """The listing a stream carries, or None for a stream of anything else (a vendor feed)."""
        listing_id = int(self._java.listingId(int(stream_id)))
        return None if listing_id == self._none else listing_id

    def listing_id_for(self, exchange_id: int, security_id: int) -> int | None:
        """The listing an order, intent or execution report names, or None if it isn't one of these."""
        listing_id = int(self._java.listingId(int(exchange_id), int(security_id)))
        return None if listing_id == self._none else listing_id

    def stream_id(self, listing_id: int) -> int | None:
        """A listing's book stream, or None."""
        stream_id = int(self._java.streamId(int(listing_id)))
        return None if stream_id == self._none else stream_id

    def listing(self, listing_id: int):
        """The registry's Java Listing (exchange, security, venue ids), or None."""
        return self._java.listing(int(listing_id))

    @property
    def raw(self):
        return self._java
