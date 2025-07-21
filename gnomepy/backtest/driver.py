import datetime
import queue

import pandas as pd

from gnomepy.backtest.event import Event, EventType
from gnomepy.backtest.exchanges import SimulatedExchange
from gnomepy.backtest.strategy import Strategy
from gnomepy.data.client import MarketDataClient
from gnomepy.data.types import SchemaType, Order, LocalMessage, CancelOrder
from gnomepy.registry.api import RegistryClient


class Backtest:

    def __init__(
            self,
            start_datetime: datetime.datetime | pd.Timestamp,
            end_datetime: datetime.datetime | pd.Timestamp,
            listing_ids: list[int],
            schema_type: SchemaType,
            strategy: Strategy,
            exchanges: dict[int, dict[int, SimulatedExchange]],
            market_data_client: MarketDataClient | None = None,
            registry_client: RegistryClient | None = None,
    ):
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.schema_type = schema_type
        self.strategy = strategy
        self.exchanges = exchanges
        self.market_data_client = market_data_client or MarketDataClient()
        self.registry_client = registry_client or RegistryClient()
        self.ready = False
        self.queue = queue.PriorityQueue()

        self._load_listings(listing_ids)
        self._verify_parameters()

    def _load_listings(self, listing_ids: list[int]):
        self.listings = []
        for listing_id in listing_ids:
            result = self.registry_client.get_listing(listing_id=listing_id)
            if not len(result):
                raise ValueError(f"Unable to find listing_id: {listing_id}")
            self.listings.append(result[0])

    def _verify_parameters(self):
        for listing in self.listings:
            if listing.exchange_id not in self.exchanges:
                raise ValueError(f"Exchange ID {listing.exchange_id} is not configured in the parameters")
            if listing.security_id not in self.exchanges[listing.exchange_id]:
                raise ValueError(f"Security ID {listing.security_id} not configured in exchange ID {listing.exchange_id}")

        for exchange_id, securities in self.exchanges.items():
            for exchange in securities.values():
                if self.schema_type not in exchange.get_supported_schemas():
                    raise ValueError(f"Exchange ID {exchange_id} does not support the provided schema type")

        for listing in self.listings:
            available_data = self.market_data_client.has_available_data(
                exchange_id=listing.exchange_id,
                security_id=listing.security_id,
                start_datetime=self.start_datetime,
                end_datetime=self.end_datetime,
                schema_type=self.schema_type,
            )
            if not available_data:
                raise ValueError(f"Listing ID {listing.listing_id} does not have data in the provided time range")

    def prepare_data(self):
        for listing in self.listings:
            records = self.market_data_client.get_data(
                exchange_id=listing.exchange_id,
                security_id=listing.exchange_id,
                start_datetime=self.start_datetime,
                end_datetime=self.end_datetime,
                schema_type=self.schema_type,
            )
            for record in records:
                self.queue.put(Event.from_schema(record))

        self.ready = True

    def execute_until(self, timestamp: int | None):
        if not self.ready:
            raise ValueError("Call prepare_data() before executing the backtest")

        while not self.queue.empty():
            event: Event = self.queue.get()

            if timestamp is not None and event.timestamp > timestamp:
                break

            if event.event_type == EventType.MARKET_DATA:
                self.exchanges[event.data.exchange_id][event.data.security_id].on_market_data(event.data)
                orders = self.strategy.on_market_data(event.data)
                for order in orders:
                    expected_timestamp = event.timestamp + self.strategy.simulate_strategy_processing_time() + \
                                         self.exchanges[order.exchange_id][order.security_id].simulate_network_latency()
                    self.queue.put(Event.from_local_message(order, expected_timestamp))
            elif event.event_type == EventType.LOCAL_MESSAGE:
                message: LocalMessage = event.data
                if isinstance(message, Order):
                    execution_report = self.exchanges[message.exchange_id][message.security_id].submit_order(message)
                elif isinstance(message, CancelOrder):
                    execution_report = self.exchanges[message.exchange_id][message.security_id].cancel_order(message)
                else:
                    raise ValueError(f"Unknown local message type: {type(message)}")

                expected_timestamp = event.timestamp + self.exchanges[message.exchange_id][message.security_id].simulate_order_processing_time() + \
                                     self.exchanges[message.exchange_id][message.security_id].simulate_network_latency()
                execution_report.timestamp_event = event.timestamp
                execution_report.timestamp_recv = expected_timestamp
                execution_report.exchange_id = message.exchange_id
                execution_report.security_id = message.security_id

                self.queue.put(Event.from_execution_report(execution_report, expected_timestamp))
            elif event.event_type == EventType.EXECUTION_REPORT:
                self.strategy.on_execution_report(event.data)
            else:
                raise ValueError(f"Unknown event type: {event.event_type}")

    def fully_execute(self):
        return self.execute_until(None)

    def persist_results(self):
        raise NotImplementedError