import datetime
import re
from typing import Optional

import boto3.session
import pandas as pd

from gnomepy.data.common import DataStore
from gnomepy.data.types import SchemaType

_KEY_REGEX = re.compile("[0-9]/[0-9]/([0-9]+)/*")

class MarketDataClient:
    def __init__(
            self,
            bucket: str = "market-data-collector",
            aws_profile_name: Optional[str] = None,
    ):
        session = boto3.session.Session(profile_name=aws_profile_name)
        self.s3 = session.client('s3')
        self.bucket = bucket

    def get_data(
            self,
            *,
            exchange_id: int,
            listing_id: int,
            start_datetime: datetime.datetime | pd.Timestamp,
            end_datetime: datetime.datetime | pd.Timestamp,
            schema_type: SchemaType
    ) -> DataStore:
        total = self._get_raw_history(exchange_id, listing_id, start_datetime, end_datetime)
        return DataStore.from_bytes(total, schema_type)

    def _get_raw_history(
            self,
            exchange_id: int,
            listing_id: int,
            start_datetime: datetime.datetime | pd.Timestamp,
            end_datetime: datetime.datetime | pd.Timestamp,
    ) -> bytes:
        keys = self._get_available_keys(exchange_id, listing_id, start_datetime, end_datetime)
        total = b''
        for key in keys:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            total += response["Body"].read()
        return total

    def _get_available_keys(
            self,
            exchange_id: int,
            listing_id: int,
            start_datetime: datetime.datetime | pd.Timestamp,
            end_datetime: datetime.datetime | pd.Timestamp,
    ):
        prefix = f"{exchange_id}/{listing_id}/"
        paginator = self.s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)

        keys = []
        for page in pages:
            for obj in page['Contents']:
                key = obj['Key']
                parsed = _KEY_REGEX.match(key)
                if parsed is not None:
                    date_hour = parsed.group(1)
                    parsed_dt = datetime.datetime.strptime(f"{date_hour}", "%Y%m%d%H")
                    if start_datetime <= parsed_dt <= end_datetime:
                        keys.append(key)

        return keys
