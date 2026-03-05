import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Generator

import boto3.session
import importlib_resources
import pandas as pd
import zstandard

from gnomepy.data.common import DataStore
from gnomepy.data.sbe import Schema
from gnomepy.data.types import SchemaBase, SchemaType, get_schema_base


class MarketDataClient:
    def __init__(
            self,
            bucket: str = "gnome-market-data-prod",
            aws_profile_name: str | None = None,
            max_workers: int = 10,
    ):
        session = boto3.session.Session(profile_name=aws_profile_name)
        self.s3 = session.client('s3')
        self.bucket = bucket
        self.max_workers = max_workers

    def get_data(
            self,
            *,
            exchange_id: int,
            security_id: int,
            start_datetime: datetime.datetime | pd.Timestamp,
            end_datetime: datetime.datetime | pd.Timestamp,
            schema_type: SchemaType,
    ) -> DataStore:
        total = self._get_raw_history(exchange_id, security_id, start_datetime, end_datetime, schema_type)
        return DataStore.from_bytes(total, schema_type)

    def stream_data(
            self,
            *,
            exchange_id: int,
            security_id: int,
            start_datetime: datetime.datetime | pd.Timestamp,
            end_datetime: datetime.datetime | pd.Timestamp,
            schema_type: SchemaType,
    ) -> Generator[SchemaBase, None, None]:
        keys = self._get_keys(exchange_id, security_id, start_datetime, end_datetime, schema_type)
        if not keys:
            return

        with importlib_resources.open_text("gnomepy.data.sbe", "schema.xml") as f:
            schema = Schema.parse(f)
        header_size = schema.types[schema.header_type_name].size()
        schema_base_type = get_schema_base(schema_type)

        # Find body_size from schema metadata
        body_size = None
        for message in schema.messages.values():
            if message.description == schema_type.value:
                body_size = message.body_size
                break
        if body_size is None:
            raise ValueError(f"Invalid schema type: {schema_type}")

        record_size = header_size + body_size
        decompressor = zstandard.ZstdDecompressor()

        with ThreadPoolExecutor(max_workers=1) as executor:
            # Submit first chunk
            next_future = executor.submit(self._fetch_chunk, keys[0])

            for i in range(len(keys)):
                chunk = next_future.result()

                # Prefetch next chunk while we iterate current one
                if i + 1 < len(keys):
                    next_future = executor.submit(self._fetch_chunk, keys[i + 1])

                if not chunk:
                    continue

                data = decompressor.decompress(chunk)
                mem = memoryview(data)
                offset = 0
                while offset + record_size <= len(mem):
                    message = schema.decode(mem[offset:])
                    parsed = schema_base_type.from_message(message)
                    yield parsed
                    offset += record_size

    def has_available_data(
            self,
            *,
            exchange_id: int,
            security_id: int,
            start_datetime: datetime.datetime | pd.Timestamp,
            end_datetime: datetime.datetime | pd.Timestamp,
            schema_type: SchemaType,
    ) -> bool:
        # TODO: Do this maybe?
        return True

    def _fetch_chunk(self, key: str) -> bytes:
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return b''

    def _get_raw_history(
            self,
            exchange_id: int,
            security_id: int,
            start_datetime: datetime.datetime | pd.Timestamp,
            end_datetime: datetime.datetime | pd.Timestamp,
            schema_type: SchemaType,
    ) -> bytes:
        keys = self._get_keys(exchange_id, security_id, start_datetime, end_datetime, schema_type)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._fetch_chunk, key) for key in keys]
            chunks = [f.result() for f in futures]

        return b''.join(chunks)

    def _get_keys(
            self,
            exchange_id: int,
            security_id: int,
            start_datetime: datetime.datetime | pd.Timestamp,
            end_datetime: datetime.datetime | pd.Timestamp,
            schema_type: SchemaType,
    ):
        keys = []
        current_time = start_datetime
        while current_time <= end_datetime:
            key = f"{security_id}/{exchange_id}/{current_time.year}/{current_time.month}/{current_time.day}/{current_time.hour}/{current_time.minute}/{schema_type.value}.zst"
            keys.append(key)
            current_time += datetime.timedelta(minutes=1)
        return keys
