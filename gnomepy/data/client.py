import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3.session
import pandas as pd
import re

from gnomepy.data.common import DataStore
from gnomepy.data.types import SchemaType

_KEY_REGEX = re.compile(r'^.*/(\d{4}/\d{1,2}/\d{1,2}/\d{1,2})/\d{1,2}/([^/]+)\.zst$')

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

    def has_available_data(
            self,
            *,
            exchange_id: int,
            security_id: int,
            start_datetime: datetime.datetime | pd.Timestamp,
            end_datetime: datetime.datetime | pd.Timestamp,
            schema_type: SchemaType,
    ) -> bool:
        keys = self._get_keys(exchange_id, security_id, start_datetime, end_datetime, schema_type)
        return len(keys) > 0

    def _get_raw_history(
            self,
            exchange_id: int,
            security_id: int,
            start_datetime: datetime.datetime | pd.Timestamp,
            end_datetime: datetime.datetime | pd.Timestamp,
            schema_type: SchemaType,
    ) -> bytes:
        keys = self._get_keys(exchange_id, security_id, start_datetime, end_datetime, schema_type)

        def fetch_key(key: str) -> bytes:
            try:
                response = self.s3.get_object(Bucket=self.bucket, Key=key)
                return response["Body"].read()
            except self.s3.exceptions.NoSuchKey:
                return b''

        chunks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_key = {executor.submit(fetch_key, key): key for key in keys}
            for future in as_completed(future_to_key):
                chunks.append(future.result())

        return b''.join(chunks)

    def _get_keys(
            self,
            exchange_id: int,
            security_id: int,
            start_datetime: datetime.datetime | pd.Timestamp,
            end_datetime: datetime.datetime | pd.Timestamp,
            schema_type: SchemaType,
    ):
        prefix = f"{security_id}/{exchange_id}/"
        paginator = self.s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)

        keys = []
        for page in pages:
            for obj in page['Contents']:
                key = obj['Key']
                parsed = _KEY_REGEX.match(key)
                if parsed is not None:
                    date_hour = parsed.group(1)
                    schema = parsed.group(2)
                    parsed_dt = datetime.datetime.strptime(f"{date_hour}", "%Y/%m/%d/%H")
                    if schema == schema_type and start_datetime <= parsed_dt <= end_datetime:
                        keys.append(key)

        return keys
