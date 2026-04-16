from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq


def resolve_fs(uri: str) -> tuple[pafs.FileSystem, str]:
    """Return (FileSystem, normalized_path) for a local path or s3:// URI."""
    if uri.startswith("s3://"):
        return pafs.FileSystem.from_uri(uri)
    return pafs.LocalFileSystem(), str(Path(uri).resolve())


def fs_exists(fs: pafs.FileSystem, path: str) -> bool:
    info = fs.get_file_info(path)
    return info.type != pafs.FileType.NotFound


def fs_mkdir(fs: pafs.FileSystem, path: str) -> None:
    if isinstance(fs, pafs.LocalFileSystem):
        Path(path).mkdir(parents=True, exist_ok=True)


def fs_read_json(fs: pafs.FileSystem, path: str) -> dict:
    with fs.open_input_file(path) as f:
        return json.loads(f.read())


def fs_write_json(fs: pafs.FileSystem, path: str, data: dict) -> None:
    content = (json.dumps(data, indent=2, default=str) + "\n").encode()
    with fs.open_output_stream(path) as f:
        f.write(content)


def fs_write_parquet(fs: pafs.FileSystem, path: str, df: pd.DataFrame) -> None:
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path, filesystem=fs)


def fs_read_parquet(fs: pafs.FileSystem, path: str) -> pd.DataFrame:
    info = fs.get_file_info(path)
    if info.type == pafs.FileType.NotFound:
        return pd.DataFrame()
    try:
        table = pq.read_table(path, filesystem=fs)
        return table.to_pandas()
    except Exception:
        return pd.DataFrame()


def fs_list_parquets(fs: pafs.FileSystem, directory: str) -> list[tuple[str, str]]:
    """Return (name_without_ext, full_path) for each .parquet file in directory."""
    try:
        selector = pafs.FileSelector(directory, recursive=False)
        result = []
        for info in fs.get_file_info(selector):
            if info.type == pafs.FileType.File and info.path.endswith(".parquet"):
                name = info.path.rsplit("/", 1)[-1][: -len(".parquet")]
                result.append((name, info.path))
        return result
    except Exception:
        return []
