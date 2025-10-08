import pandas as pd
from typing import Literal, Any
from gnomepy.backtest.stats.metric import Metric


IntervalType = Literal["second", "minute", "hour", "day", "month", "year"]


def compute_metrics(
        df: pd.DataFrame,
        metrics: list[Metric],
) -> dict[str, Any]:
    context = {
        'start_timestamp': df.index.min(),
        'end_timestamp': df.index.max()
    }

    for metric in metrics:
        context[metric.name] = metric.compute(df=df, ctx=context)

    return context


def compute_metrics_for_intervals(
        df: pd.DataFrame,
        metrics: list[Metric],
        interval: IntervalType | None,
) -> list[dict[str, Any]]:
    partitions = [df]

    if interval is not None:
        partitions.extend(partition(df, interval))

    return [compute_metrics(sub_df, metrics) for sub_df in partitions]


def resample(
        df: pd.DataFrame,
        frequency: str,
        aggregate: dict[str, str] | None = None,
        **kwargs,
):
    if aggregate is None:
        # aggregate = {"price": "mean", "quantity": "sum", "fee": "sum", "nmv": "mean"}
        aggregate = {"event": "mean", "price": "mean", "quantity": "mean", "fee": "sum"}

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("Index dtype of datetime is expected when resampling.")

    return df.resample(frequency, **kwargs).agg(aggregate)


def partition(
        df: pd.DataFrame,
        interval: IntervalType
) -> list[pd.DataFrame]:
    """Split a time-indexed DataFrame into daily or monthly groups."""
    dt_format = interval_to_dt_format(interval)
    return [grp for _, grp in df.groupby(df.index.strftime(dt_format))]


def interval_to_dt_format(interval: str) -> str:

    if interval == "second":
        return "%Y%m%d:%H%M%S"
    elif interval == "minute":
        return "%Y%m%d:%H%M"
    elif interval == "hour":
        return "%Y%m%d:%H"
    elif interval == "day":
        return "%Y%m%d"
    elif interval == "month":
        return "%Y%m"
    elif interval == "year":
        return "%Y"
    else:
        raise ValueError(f"Unknown interval: {interval}")


