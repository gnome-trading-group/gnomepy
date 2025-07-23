import datetime
import re

import boto3.session
import pandas as pd

from gnomepy.data.common import DataStore
from gnomepy.data.types import SchemaType

_KEY_REGEX = re.compile("^\\d+/\\d+/(\\d{10})/([^/]+)\\.zst$")

class MarketDataClient:
    def __init__(
            self,
            bucket: str = "gnome-market-data-prod",
            aws_profile_name: str | None = None,
    ):
        session = boto3.session.Session(profile_name=aws_profile_name)
        self.s3 = session.client('s3')
        self.bucket = bucket

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
        # TODO: Do this maybe?
        return True

    def _get_raw_history(
            self,
            exchange_id: int,
            security_id: int,
            start_datetime: datetime.datetime | pd.Timestamp,
            end_datetime: datetime.datetime | pd.Timestamp,
            schema_type: SchemaType,
    ) -> bytes:
        keys = self._get_available_keys(exchange_id, security_id, start_datetime, end_datetime, schema_type)
        total = b''
        for key in keys:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            total += response["Body"].read()
        return total

    def _get_available_keys(
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
                    parsed_dt = datetime.datetime.strptime(f"{date_hour}", "%Y%m%d%H")
                    if schema == schema_type and start_datetime <= parsed_dt <= end_datetime:
                        keys.append(key)

        return keys
