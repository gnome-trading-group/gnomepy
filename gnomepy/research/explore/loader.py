import pandas as pd

from gnomepy.data.cached_client import CachedMarketDataClient
from gnomepy.data.types import SchemaType
from gnomepy.research.data.mbp10.features import (
    compute_microstructure_bulk,
    compute_returns_bulk,
    compute_trade_flow_bulk,
)

_FEATURE_GROUPS = {
    "microstructure": compute_microstructure_bulk,
    "returns": compute_returns_bulk,
    "trade_flow": compute_trade_flow_bulk,
}


def load_raw_mbp10(
    exchange_id: int,
    security_id: int,
    start,
    end,
    *,
    client: CachedMarketDataClient | None = None,
) -> pd.DataFrame:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if client is None:
        client = CachedMarketDataClient()
    ds = client.get_data(
        exchange_id=exchange_id,
        security_id=security_id,
        start_datetime=start,
        end_datetime=end,
        schema_type=SchemaType.MBP_10,
    )
    df = ds.to_df(price_type="float", size_type="float", pretty_ts=True)
    df["midPrice"] = (df["bidPrice0"] + df["askPrice0"]) / 2.0
    return df


def load_mbp10(
    exchange_id: int,
    security_id: int,
    start,
    end,
    *,
    features: list[str] | None = None,
    client: CachedMarketDataClient | None = None,
) -> pd.DataFrame:
    df = load_raw_mbp10(exchange_id, security_id, start, end, client=client)
    if features is None:
        features = list(_FEATURE_GROUPS.keys())
    for name in features:
        fn = _FEATURE_GROUPS[name]
        feat_df = fn(df)
        df = df.join(feat_df, how="left")
    return df
