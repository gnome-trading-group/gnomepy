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
