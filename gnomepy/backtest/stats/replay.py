from __future__ import annotations

import pandas as pd

from gnomepy.backtest.stats.stats import BaseRecord


def align_records(records: dict[str, BaseRecord]) -> pd.DataFrame:
    """Outer-join multiple named records on their timestamp index.

    Each record's DataFrame columns are prefixed with its name so that
    columns from different recorders never collide.  After the join,
    numeric columns are forward-filled to handle differing recording
    frequencies.

    Parameters
    ----------
    records : dict[str, BaseRecord]
        Mapping of label → ``BaseRecord`` instance.  The records may come
        from any recorder type (market, intent, custom).

    Returns
    -------
    pd.DataFrame
        A single DataFrame whose index is the union of all record
        timestamps.  Each column is named ``{label}.{original_column}``.
    """
    frames = []
    for name, record in records.items():
        df = record.df.copy()
        df.columns = [f"{name}.{col}" for col in df.columns]
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    result = frames[0]
    for df in frames[1:]:
        result = result.join(df, how='outer')

    numeric_cols = result.select_dtypes(include=['number']).columns
    result[numeric_cols] = result[numeric_cols].ffill()

    return result


def compare_runs(
    runs: dict[str, 'Recorder'],
    listing_id: int,
    recorder_names: list[str] | None = None,
) -> pd.DataFrame:
    """Outer-join recorder data from multiple backtest runs on a shared timestamp axis.

    For each run, the specified recorders are extracted and their columns
    are prefixed with ``{run_name}.{recorder_name}.{column}``.

    Parameters
    ----------
    runs : dict[str, Recorder]
        Mapping of run label → ``Recorder`` from a completed backtest.
    listing_id : int
        The listing whose records to compare across runs.
    recorder_names : list[str] | None, default None
        Which recorders to include.  ``None`` means all: ``'market'`` plus
        every registered custom recorder.

    Returns
    -------
    pd.DataFrame
        A single DataFrame whose index is the union of all timestamps
        across all runs and recorders.  Numeric columns are forward-filled.
    """
    from gnomepy.backtest.recorder import Recorder

    frames = []
    for run_name, recorder in runs.items():
        to_extract: dict[str, BaseRecord] = {}

        names = recorder_names if recorder_names is not None else ['market'] + list(recorder.custom_recorders.keys())

        for rec_name in names:
            if rec_name == 'market':
                record = recorder.market_recorder.get_record(listing_id)
            elif rec_name == 'intent':
                record = recorder.intent_recorder.get_record(listing_id)
            elif rec_name in recorder.custom_recorders:
                record = recorder.custom_recorders[rec_name].get_record(listing_id)
            else:
                continue

            if len(record.arr) == 0:
                continue

            df = record.df.copy()
            df.columns = [f"{run_name}.{rec_name}.{col}" for col in df.columns]
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    result = frames[0]
    for df in frames[1:]:
        result = result.join(df, how='outer')

    numeric_cols = result.select_dtypes(include=['number']).columns
    result[numeric_cols] = result[numeric_cols].ffill()

    return result
