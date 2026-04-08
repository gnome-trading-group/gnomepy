# backtest image

Runs `gnomepy backtest` against a pinned commit of the private
`gnomepy-research` repo. Java JARs (`gnome-backtest`) are baked into
the image; the research repo is cloned at run time so a single image
can target any commit.

## Build

The build context must contain **both** `gnomepy` and `gnome-backtest`
sibling checkouts. From the parent of those repos:

The Maven step needs GitHub Packages credentials to resolve the
`gnome-parent` POM. Export `GITHUB_ACTOR` (your GH username) and
`GITHUB_TOKEN` (a PAT with `read:packages`) and pass them in:

```sh
DOCKER_BUILDKIT=1 docker build \
  --build-arg GITHUB_ACTOR="$GITHUB_ACTOR" \
  --secret id=github_token,env=GITHUB_TOKEN \
  -f gnomepy/Dockerfile.research-backtest \
  -t gnomepy-research-backtest \
  .
```

## Run

```sh
docker run --rm \
  -e RESEARCH_COMMIT=<git-sha> \
  -e GH_TOKEN=$GH_TOKEN \
  -e BACKTEST_CONFIG=/work/backtest.yaml \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_SESSION_TOKEN \
  -v "$PWD/backtest.yaml:/work/backtest.yaml:ro" \
  -v "$PWD/out:/work/out" \
  gnomepy-research-backtest \
  --output /work/out/results.json
```

Required env:

| var             | meaning                                                     |
| --------------- | ----------------------------------------------------------- |
| `RESEARCH_COMMIT` | git SHA (or ref) of `gnomepy-research` to check out           |
| `GH_TOKEN`        | GitHub token with read access to `gnomepy-research`           |
| `BACKTEST_CONFIG` | path (inside container) to a YAML backtest config           |

Optional: `AWS_*` for the S3 market-data bucket, plus any extra
`gnomepy backtest` flags as positional args (e.g. `--output ...`).

A minimal sample config lives at `example-backtest.yaml`.

## Notes

- `gnomepy` itself is installed from the working tree at build time,
  so the image pins one specific gnomepy version. Rebuild the image to
  bump it.
- `gnomepy-research` is installed with `--no-deps` so it cannot pull a
  different `gnomepy` over the baked-in one.
- `GH_TOKEN` is unset before exec'ing the backtest.
