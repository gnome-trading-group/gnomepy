"""Interactive backtest explorer powered by Plotly Dash.

Usage::

    gnomepy explore ./path/to/backtest
    gnomepy explore ./backtest_a --compare ./backtest_b
"""
from __future__ import annotations

import threading
import webbrowser
from typing import TYPE_CHECKING

from gnomepy.explorer.app import create_app
from gnomepy.explorer.data import ComparisonStore, ExplorerDataStore

if TYPE_CHECKING:
    from gnomepy.java.recorder import BacktestResults


def launch_explorer(
    results_a: BacktestResults,
    results_b: BacktestResults | None = None,
    port: int = 8050,
    open_browser: bool = True,
    debug: bool = False,
    price_decimals: int = 2,
) -> None:
    """Load results and launch the Dash backtest explorer in a local browser."""
    import time

    def _step(msg: str) -> float:
        print(f"  {msg}", flush=True)
        return time.time()

    print("Loading backtest data…", flush=True)
    t0 = time.time()

    _step("reading parquet files…")
    store_a = ExplorerDataStore(results_a, label="A", _progress=_step)
    store_b = ExplorerDataStore(results_b, label="B", _progress=_step) if results_b else None

    print(f"  ready in {time.time() - t0:.1f}s — {store_a.summary_label()}", flush=True)
    if store_b:
        print(f"  B: {store_b.summary_label()}", flush=True)

    app = create_app(store_a, store_b, price_decimals=price_decimals)

    if open_browser:
        url = f"http://localhost:{port}"
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
        print(f"Opening {url} …")

    print(f"Explorer running at http://localhost:{port}")
    app.run(port=port, debug=debug)
