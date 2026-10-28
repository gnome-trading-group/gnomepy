# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
poetry install          # install dev environment
poetry run pytest       # run all tests
poetry run pytest tests/test_sweep.py   # run a single test file
poetry build            # build wheel + sdist
scripts/release.sh patch|minor|major   # from main: bump, commit, tag vX.Y.Z, push; the tag publishes to PyPI
```

No lint/format tooling is configured — no black, ruff, or mypy.

CLI (after `poetry install`):
```bash
gnomepy backtest run --config config.yaml       # local backtest
gnomepy backtest submit --config config.yaml    # submit to AWS Batch
gnomepy backtest status <run_id>               # check remote run
gnomepy login                                   # OAuth2 PKCE auth flow
```

## Architecture

**Gnomepy is a Python/JVM bridge for backtesting market-making strategies.** Python strategy code calls into a high-performance Java simulation engine via JPype.

### Key modules

- `java/backtest/runner.py` — `Backtest` class and `run_backtest()`: the main orchestration entry point. Starts JVM, builds Java config objects, wires strategy callbacks, runs replay.
- `java/backtest/strategy.py` — `Strategy` base class. Subclass and implement `on_market_data` and `on_execution_report`. Python strategies are wrapped into JPype `@JImplements` proxies so Java can call them.
- `java/backtest/config.py` — Dataclasses (`BacktestConfig`, `SimulationOverrides`, `RegistryFeeConfig`/`StaticFeeConfig`/`ParametricFeeConfig`, etc.) that mirror Java config objects. Each has a `_to_java()` method. Each listing simulates with its exchange's registry simulation parameters; `BacktestConfig.exchanges` (by exchange code) and `ListingSimConfig.overrides` replace blocks of them, and `simulation_as_of` pins which version a run reads.
- `java/recorder.py` — `MetricRecorder` and `BacktestResults`. Strategies write custom tick-level signals to named buffers; results land in parquet files.
- `java/_jvm.py` — JVM singleton lifecycle. One JVM per process, lazy-started via `ensure_jvm_started()`, auto-shutdown via atexit. Classpath comes from `java/_classpath.py`: the `GNOME_JARS` env var (comma-separated jar paths), else `GNOME_ROOT/gnome-backtest/target/*-all.jar`, else a sibling `gnome-backtest` checkout.
- `reporting/` — `BacktestReport` (HTML export), `metrics.py` (vectorized PnL/Sharpe/curves), `plots.py` (Plotly sections).
- `sweep.py` — Expands YAML configs with list or `{min,max,step}` range values into a cartesian product for parameter sweeps. Also supports a `scenarios` key for running the same strategy across multiple listing/timewindow combinations; total jobs = scenarios × sweep combinations.
- `remote.py` — HTTP client wrapping the gnome-controller backtest and research job APIs (submit, status, list, cancel; research job input uploads and output downloads). Uses Cognito id_token from `auth.py`. The research job runner image is `docker/research-job/`.
- `registry/` — `RegistryClient` for the gnome-registry API (security master, streams/sources, fees, simulation parameters, strategies/sessions, alerts). See "Registry client" below.
- `config.py` — Per-stage hosts (`DevConfig`/`ProdConfig`: registry, market-data API, controller, Cognito) picked by the `STAGE` env var, plus API key and alerts URL resolution.

### Data flow

```
YAML config
    └─► sweep.py (scenario × sweep expansion) ─► N individual configs
                                               │
                                        Backtest.run()
                                               │
                              ┌────────────────┼─────────────────┐
                              │                │                  │
                       Strategy proxy    Java BacktestConfig   JVM engine
                       (JPype callback)  (via _to_java())      replays ticks
                              │                                   │
                       on_market_data / on_execution_report ◄─────┘
                              │
                       emit Intent list ──► OMS processes orders
                              │
                       MetricRecorder ──► parquet + metadata.json
                              │
                       BacktestReport ──► report.html + summary.json
```

### Critical invariants

**JVM:** One global JVM per process. Never start it manually — call `ensure_jvm_started()`. JPype keyword collisions (e.g. Java reserved words as field names) require the `"NONE"` string workaround.

**JVM flags live only in `java/_jvm.py`.** `DEFAULT_JVM_ARGS` (backtests) and `STRATEGY_JVM_ARGS` (live/paper sessions) are the single source; gnome-orchestrator's EC2 `start-session.sh` reads `STRATEGY_JVM_ARGS` from the installed gnomepy. Never copy the flags into a script or Dockerfile, or the Python and Java launch paths drift apart.

**Backtest jars must match.** gnomepy runs whatever Java is on the classpath, normally the gnome-backtest uber jar, which bundles the gnome-oms version pinned in gnome-backtest's `pom.xml`. After a gnome-oms (or gnome-simulation) change, bump that pin and rebuild gnome-backtest (`mvn install -DskipTests`), or `GNOME_JARS` points at an engine without the change. If `GNOME_JARS` lists separate jars, their versions must agree.

**Prices/quantities are scaled integers.** Divide raw values by `Scales.PRICE` (1e9) for display. Never pass floats directly to Java intent encoders.

**Intent pattern:** Strategies declare desired state (quotes/takes) as `Intent` objects. The OMS diffs intents against live orders — strategies do not manage order IDs directly.

**Schema wrappers** (`java/schemas.py`) are thin Python views over live Java SBE objects. Properties decode on access; don't cache the wrapper across ticks.

**MetricRecorder buffers** must be created and frozen (column schema fixed) before the replay starts. Rows are appended tick-by-tick during replay.

**Sweep simulation blocks:** Dot-notation keys like `exchanges.KALSHI.fee_model.taker_fee` or `listings.48212.network_latency.latency_nanos` target nested YAML fields; the swept block must name its model (`type`) in the config. Multiple keys produce a cartesian product with strategy args.

**Missing market data:** a backtest asks the market-data API's `collection/missing` (Java `MarketDataApi`, built in `runner.py` with `MARKET_DATA_API_KEY`, else the `gnome/api-keys/market-data` mirror) where collection had no data for its streams. `on_missing` (config): `mark` (default) re-sends the stream's last book flagged bad data at each stretch's start, as the gateway does live; `fail` refuses the run; `ignore` replays as recorded. Every run reports it in `metadata.market_data` and `summary.json`'s `market_data_missing_fraction` (the controller's "Missing data" column); `report.html` shades the stretches on its time charts (hover the ▾ markers) and lists them in a Market Data section; without a key the run warns and reports nothing. Time before coverage was tracked is `untracked_ns`, never marked.

**Scenarios:** A `scenarios` key at the top level defines named listing/timewindow groups (each with `start_date`, `end_date`, `listings`, and optional `strategy_args` overrides). When present, `start_date`/`end_date`/`listings` are omitted from the top level. Total jobs = scenarios × sweep combinations; results are tagged with scenario name.

## Registry client

`gnomepy/registry/api.py` — `RegistryClient` wraps the gnome-registry REST API. Reads page through `limit`/`offset` and parse into the dataclasses in `registry/types.py`.

- **Host:** `https://{config.REGISTRY_API_HOST}/api`, chosen by `STAGE` (`dev` or `prod`; unset or unknown means prod, so set `STAGE=dev` when testing against dev). `base_url=` overrides it.
- **API key:** `api_key=`, else `GNOME_REGISTRY_API_KEY`, else `REGISTRY_API_KEY_ID` looked up through API Gateway `GetApiKey`, else the Secrets Manager mirror `gnome/api-keys/registry` (both need AWS credentials; Batch backtest and research job roles may read the mirror). Lookups are cached per process because `GetApiKey` is an account-wide, tightly throttled call and warm Lambdas build a client per invocation. No key raises `ValueError`. A stale `GNOME_REGISTRY_API_KEY` in a shell (e.g. prod's key with `STAGE=dev`) wins over the mirror and gets 401s.
- **Operator routes:** `_post_as_operator`/`_patch_as_operator` send the `gnomepy login` Cognito ID token (`auth.get_id_token()`) instead of the API key, because those routes must name a person in the audit log (e.g. `stop_strategy_session`, `create_vendor_source`, `update_source`). They fail in Lambdas and CI where nobody has logged in; automated callers use the API-key variant (e.g. `stop_strategy_session_as_service` with an `actor`).
- **Alerts:** `send_alert` posts to `<alerts url>/alerts` with the client's API key. `type` is a type from the registry's alert catalog (e.g. `LAUNCH_FAILED`) or `CUSTOM`, whose `severity` the caller sets (capped below paging). Reuse a `key` to dedupe while firing and send `status="RESOLVED"` with it to close. The URL is `alerts_url=`, else `ALERTS_API_URL` (sessions have it), else SSM `/gnome/alerting/api-url`.
- Request timeout is 60s: API Gateway gives up at 29s, so anything longer is a dropped connection.

**Parsed types must tolerate registry drift.** `_get` already drops response keys a dataclass doesn't know, so the registry adding a field is safe. A field the registry stops returning is not: a dataclass field without a default raises `TypeError` on every read (the registry stopped returning `Exchange.schema_type`, so `get_exchange` fails on every gnomepy before 3.1.0, which removed the field; consumers still on 2.x are exposed). Give new or optional fields a default, remove a field here before or with its removal from the registry, and test parsing against a response missing it.

## Downstream consumers

Many services install gnomepy from PyPI, so a release that breaks a public API or a parsed type breaks each of them until it bumps gnomepy (or keeps failing on a floating range):

- gnome-classifier (`gnomepy >3.0.0`, floats, so it picks up a new release on its next build)
- gnome-launcher (`^3.1.0`), gnome-bot (`^3.1.0`)
- gnomepy-research (`>=3.1.0,<4.0.0`), and through it research jobs and backtests
- gnome-pipelines (`gnomepy[live-data] >=3.1.0,<4.0.0`, and `==3.1.0` in its data-api image, kept equal to its lock)
- Live/paper strategy sessions: gnome-orchestrator's EC2 `start-session.sh` installs `gnomepy[strategy,live-data]==$GNOMEPY_VERSION` (set by the registry's session launcher from the session config's `gnomepyVersion`, else the latest gnomepy on PyPI at launch) and reads `STRATEGY_JVM_ARGS` from it. So an unpinned session picks up a new gnomepy release on its next launch.

Release a fix with `scripts/release.sh`, then bump each affected consumer. Majors (`3.0.0`) need consumers with caret or `<N` ranges to bump deliberately.

## Operations

- **Releases:** `scripts/release.sh patch|minor|major` must run on a clean `main` in sync with `origin/main`. It bumps `pyproject.toml` with Poetry, commits `release: vX.Y.Z`, tags it, asks before pushing, then pushes commit and tag atomically. The tag triggers `.github/workflows/deploy.yml`, which checks the tag matches the package version, builds with Poetry and publishes to PyPI. Releasing and pushing is the user's call.
- **Docker images** (both built by GitHub Actions on push to `main` touching `gnomepy/**`, their `docker/` dir, `pyproject.toml` or `poetry.lock`, and pushed with tags `latest` and the gnomepy short SHA to ECR in us-east-1 in both dev and prod):
  - `gnomepy-backtest` (`docker/backtest/`, workflow `build-backtest-image.yml`): gnomepy from this tree plus the gnome-backtest uber jar built from gnome-backtest `main`. Also rebuilt when gnome-backtest's release workflow sends the `gnome-backtest-updated` dispatch. Runs `gnomepy backtest run` on AWS Batch for `gnomepy backtest submit` (gnome-controller `backtest-stack.ts`).
  - `gnomepy-research-job` (`docker/research-job/`, workflow `build-research-job-image.yml`, also `workflow_dispatch`): gnomepy from this tree plus gnomepy-research's locked dependencies, no JVM. Runs research job tasks on AWS Batch (gnome-controller `research-jobs-stack.ts`).
  - Local builds: `docker/backtest/build.sh` (needs `GITHUB_ACTOR` and `GITHUB_TOKEN` with `read:packages`) and `docker/research-job/build.sh`; both expect sibling checkouts next to `gnomepy`.
- The Batch job definitions run the `latest` tag, so a push to `main` changes what every new backtest and research job runs in dev and prod immediately, before any PyPI release. A broken `main` breaks remote backtests and research jobs right away.
- **CI:** only the three workflows above. Tests do not run in CI; run `poetry run pytest` before pushing.
