# Backtest Explorer

The backtest explorer is an interactive Dash app for forensic analysis of completed backtests. It gives you tick-level visibility into what your strategy was doing, lets you step through fills one by one, and supports side-by-side comparison of two runs.

## Launching

### From a local path

```bash
gnomepy explore ./019e2174-4c28-7923-9855-a59835605e95
```

### From an S3 URI

```bash
gnomepy explore s3://gnome-research-prod/backtests/<run_id>/jobs/0
```

### From a remote run_id

The easiest way for remote backtests. Looks up the S3 path automatically:

```bash
gnomepy explore --run-id <run_id>
gnomepy explore --run-id <run_id> --job 3   # specific sweep job
```

You can also copy these commands directly from the gnome-controller UI — they appear in the run card and in an "Explore" column on each succeeded job row.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--run-id` | — | Remote run ID (alternative to PATH) |
| `--job N` | `0` | Job index within a sweep run |
| `--compare PATH_OR_RUN_ID` | — | Second backtest for comparison mode |
| `--compare-job N` | `0` | Job index for the comparison run |
| `--port N` | `8050` | Local server port |
| `--no-browser` | off | Skip auto-opening a browser tab |

---

## The UI

The explorer opens in your browser at `http://localhost:8050`. All four panels share the same time axis — zooming any chart zooms all of them.

### Navigation bar

```
[|◀] [◀] [▶] [▶] [▶|]   Event: [Fills ▼]   Speed: [1× ▼]   Signals: [fair_value ▼]   [Reset]
[══════════|══════════════════════════════] ← range slider
Window: 2026-01-21 03:14:22 — 03:15:22  (60.0s)
```

- **Step buttons** (`|◀ ◀ ▶ ▶|`): jump to previous/next event of the selected type
- **Play/Pause**: auto-advance through events at 1×, 5×, or 10× speed
- **Event filter**: choose which event type to step through — Fills, Intents, Orders, or All
- **Range slider**: coarse navigation across the full session; drag handles to set the view window
- **Reset**: zoom back out to the full session

### Panel 1 — Price & Book

The flagship panel. Shows:

- **Mid price** (solid blue line)
- **BBO band** (green/red shaded area between best bid and ask)
- **Fill markers**: green triangles (buys) and red triangles (sells) pinned to fill price. Hover to see price, qty, fee, and slippage in bps.
- **Strategy quotes** (dashed lines): bid and ask prices from your strategy's intents — lets you see what the strategy was quoting vs where it actually filled
- **Take price** (dashdot line): when the strategy was taking liquidity

If your backtest was recorded with `record_depth > 1`, zooming into a window narrower than 60 seconds switches the BBO band to a full depth visualization showing all recorded price levels.

### Panel 2 — Signals

Only appears if your strategy recorded custom metrics via `MetricRecorder`. Use the **Signals** dropdown in the nav bar to select which columns to plot. Each signal is a line trace; in comparison mode, run A is solid and run B is dashed.

### Panel 3 — PnL & Position

Two synchronized subplots:

- **PnL** (top): mark-to-market PnL computed as `cash + position × mid`. In comparison mode, shows run A solid, run B dashed, and a filled Δ area (A − B).
- **Position** (bottom): running net position as a step line. Flat = you're flat, positive = long, negative = short.

### Panel 4 — Event Log

A filterable, sortable table of all events in the current time window:

| Time | Type | Src | Side | Price | Qty | Fee | Slip bps | Status |
|------|------|-----|------|-------|-----|-----|----------|--------|
| 03:14:22.005 | fill | A | Bid | 97842.123456 | 0.0100 | 0.000098 | −1.24 | — |
| 03:14:22.011 | intent | A | Bid/Ask | 97841.900000 | 0.0200 | — | — | — |

**Click any row to jump the view to that timestamp.** All panels immediately re-center on that event.

- **fill**: an actual execution — price, qty, fee, and slippage vs mid at fill time
- **intent**: a strategy quote update — what the strategy wanted to do
- **order**: an order in terminal state — shows `final_status` (Filled, Cancelled, Rejected, etc.)
- **Src** column (A/B): only shown in comparison mode

---

## Comparison mode

Load two backtests side by side:

```bash
# Two run IDs
gnomepy explore --run-id <run_a> --compare <run_b>

# Two specific sweep jobs
gnomepy explore --run-id <run_a> --job 2 --compare <run_b> --compare-job 5

# Mix local and remote
gnomepy explore ./local-run --compare s3://gnome-research-prod/backtests/<run_b>/jobs/0
```

In comparison mode:

- **Price panel**: fills from run A as triangles, run B as circles. Same mid price (shared market data).
- **PnL panel**: both PnL curves overlaid, plus a Δ fill (A − B) showing where one run outperformed the other.
- **Signals panel**: run A solid, run B dashed in the same color.
- **Event log**: merged events from both runs with a "Src" column (A or B).
- **Config diff**: if the two runs have different strategy args, a collapsible "Config Differences" accordion appears in the header.

---

## Workflows

### Finding why a trade happened

1. In the Event filter, select **Fills**
2. Press **▶** to step to the next fill
3. Look at the price panel: was the fill at a wide spread or tight spread? Was the strategy quoting aggressively?
4. Look at the event log: what intent was posted just before this fill?
5. Look at the signals panel: was your fair value above/below mid at fill time?

### Diagnosing a drawdown period

1. In the PnL panel, identify a drawdown visually
2. Click on the start of the drawdown in the PnL panel — the view zooms to that timestamp
3. Zoom in further using scroll wheel or box zoom
4. Press **▶** repeatedly to walk through each fill during the drawdown
5. Check slippage values in the event log — consistently negative slippage means adversely selected

### Comparing two hyperparameter values

1. Run two backtests with different parameter values, note their run IDs
2. `gnomepy explore --run-id <a> --compare <b>`
3. Look at the PnL Δ chart — areas where the delta widens show when the configurations diverged
4. Zoom into a divergence period and look at the event log: does run A have fills that run B doesn't?
5. Use the step buttons with **All events** selected to walk through the moments where one run traded and the other didn't

### Checking execution quality

Filter to **Fills** and look at the **Slip bps** column in the event log:

- Negative values: you filled better than mid (typical for maker fills)
- Positive values: you paid through mid (typical for taker fills or adverse selection)
- Large positive values on maker fills: the market moved against you at the moment of execution

Zoom into a fill on the price panel to see the BBO at the exact fill timestamp versus your fill price.
