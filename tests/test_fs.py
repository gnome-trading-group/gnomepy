import gzip

import pyarrow.fs as pafs

from gnomepy._fs import fs_read_bytes, fs_write_bytes, fs_write_json


def test_bytes_round_trip_unchanged_whatever_the_suffix(tmp_path):
    fs = pafs.LocalFileSystem()
    page = gzip.compress(b"<html><footer></footer></html>")

    for name in ("page.html.gz", "data.zst", "plain.bin"):
        fs_write_bytes(fs, str(tmp_path / name), page)
        assert fs_read_bytes(fs, str(tmp_path / name)) == page


def test_json_is_written_as_text_even_under_a_compressed_suffix(tmp_path):
    fs = pafs.LocalFileSystem()

    fs_write_json(fs, str(tmp_path / "summary.json.gz"), {"a": 1})

    assert (tmp_path / "summary.json.gz").read_bytes().startswith(b"{")
