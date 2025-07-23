from pathlib import Path
import datetime
import pandas as pd

from gnomepy.data.client import MarketDataClient
from gnomepy.data.common import DataStore, CachedDataStore
from gnomepy.data.types import SchemaType


class CachedMarketDataClient(MarketDataClient):
    def __init__(
        self,
        bucket: str = "gnome-market-data-prod",
        aws_profile_name: str | None = None,
        local_cache_dir: str = "/tmp/market_data_cache",
    ):
        super().__init__(bucket=bucket, aws_profile_name=aws_profile_name)
        self.local_cache_dir = Path(local_cache_dir)

    def _local_cache_path(self, key: str) -> Path:
        return self.local_cache_dir / self.bucket / key

    def _local_data_store_cache_path(
            self,
            exchange_id: int,
            security_id: int,
            start_datetime: datetime.datetime | pd.Timestamp,
            end_datetime: datetime.datetime | pd.Timestamp,
            schema_type: SchemaType,
    ):
        return self.local_cache_dir / str(security_id) / str(exchange_id) / start_datetime.isoformat() / end_datetime.isoformat() / schema_type.name

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
        cache_path = self._local_data_store_cache_path(exchange_id, security_id, start_datetime, end_datetime, schema_type)
        return CachedDataStore.from_bytes(total, schema_type, cache_path=cache_path)

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
            local_path = self._local_cache_path(key)

            if local_path.exists():
                with local_path.open("rb") as f:
                    data = f.read()
            else:
                response = self.s3.get_object(Bucket=self.bucket, Key=key)
                data = response["Body"].read()

                local_path.parent.mkdir(parents=True, exist_ok=True)
                with local_path.open("wb") as f:
                    f.write(data)

            total += data

        return total
