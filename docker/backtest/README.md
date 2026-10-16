# backtest image

Runs `gnomepy backtest` against a pinned commit of the private
`gnomepy-research` repo. Java JARs (`gnome-backtest`) are baked into
the image; the research repo is cloned at run time so a single image
can target any commit.

## Build

The build context must contain **both** `gnomepy` and `gnome-backtest`
sibling checkouts. Export `GITHUB_ACTOR` (your GH username) and
`GITHUB_TOKEN` (a classic PAT with `read:packages`), then run:

```sh
./gnomepy/docker/backtest/build.sh
```

Optional: `IMAGE_TAG`, `GNOME_BACKTEST_REF` (git ref to clone & build
instead of the local sibling checkout).

## Runtime (AWS Batch)

The entrypoint expects these env vars (set by the Batch job definition):

| var               | meaning                                           |
| ----------------- | ------------------------------------------------- |
| `RUN_ID`          | backtest run identifier                           |
| `S3_BUCKET`       | S3 bucket for configs and results                 |
| `RESEARCH_COMMIT` | git SHA or ref of `gnomepy-research` to check out |

`AWS_BATCH_JOB_ARRAY_INDEX` is injected automatically by Batch (defaults to 0).
`GH_TOKEN` is fetched from Secrets Manager (`gnomepy/gh-token`) at startup.

## Notes

- `gnomepy` is installed from the working tree at build time — rebuild the image to bump it.
- `gnomepy-research` is installed with `--no-deps` so it cannot override the baked-in `gnomepy`.
- `GH_TOKEN` is unset before the backtest runs.

## Example configs

`example-backtest.yaml` documents every top-level option; `examples/` holds runnable ones:

| config                              | shows                                                        |
| ----------------------------------- | ------------------------------------------------------------ |
| `momentum_signal_sweep.yaml`        | sweeping strategy args                                       |
| `momentum_fee_threshold_sweep.yaml` | sweeping an exchange's fees alongside strategy args          |
| `momentum_latency_sweep.yaml`       | sweeping an exchange's network latency                       |
| `momentum_queue_model_sweep.yaml`   | sweeping an exchange's queue model                           |
| `multi_scenario_sweep.yaml`         | several events in one run, crossed with a sweep              |
| `listing_overrides.yaml`            | overriding simulation for one exchange and for one listing   |

Each listing simulates with its exchange's simulation parameters from the registry (controller → Security
Master → Simulation) and pays its venue's fees from the registry's fee schedules. A config sets only what an
experiment changes: `exchanges.<CODE>` for every listing on an exchange, then `listings[].overrides` for one listing,
each replacing whole blocks of the layer below.
