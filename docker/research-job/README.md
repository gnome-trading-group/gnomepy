# research-job image

Runs one task of a research job: `python -m <module> <args>` from a pinned commit of the private
`gnomepy-research` repo, on AWS Batch (gnome-controller `ResearchJobsStack`, queue `gnome-research-queue`).
gnomepy and every gnomepy-research dependency family (pymc/nutpie, xgboost, websocket, research) are baked
in at gnomepy-research's locked versions; the research repo is cloned at run time so one image serves any
commit. There is no JVM: backtests have their own image (`docker/backtest`).

## Build

The build context must contain **both** `gnomepy` and `gnomepy-research` sibling checkouts:

```sh
./gnomepy/docker/research-job/build.sh
```

Optional: `IMAGE_TAG`. CI (`.github/workflows/build-research-job-image.yml`) builds on every push to gnomepy
main that touches the package or this directory, and on demand (workflow_dispatch), and pushes `latest` and the
gnomepy short SHA to the `gnomepy-research-job` ECR repo in dev and prod. Rebuild after gnomepy-research
changes its locked dependencies: a task installs its commit against `constraints.txt` anyway, so a stale image
still works, it just spends time installing what changed.

## Runtime

`run_task.py` is the entrypoint. Env:

| var                         | set by                         | meaning                                       |
| --------------------------- | ------------------------------ | --------------------------------------------- |
| `RUN_ID`                    | submit Lambda                  | research job run                              |
| `RESEARCH_COMMIT`           | submit Lambda                  | full SHA of `gnomepy-research` to check out   |
| `S3_BUCKET`                 | job definition                 | the research bucket                           |
| `AWS_DEFAULT_REGION`        | job definition                 |                                               |
| `AWS_BATCH_JOB_ARRAY_INDEX` | Batch (array jobs)             | task index; else `JOB_INDEX`, else 0          |

For task `i` of run `r` it:

1. reads `research-jobs/runs/r/jobs/i/task.json` (`{module, args, inputs}`)
2. downloads each input set (`research-jobs/inputs/<sha256>/`) to `/work/in/<name>/`
3. clones `gnomepy-research` at the commit and installs it with all its extras, pinned by its
   `constraints.txt` and by the image's gnomepy version (a commit that needs a newer gnomepy fails here, by name)
4. runs `python -m <module> <args...>` with `OUTPUT_DIR=/work/out`, `INPUT_DIR=/work/in`, `JOB_INDEX=i`
5. always uploads `/work/out/` to `research-jobs/runs/r/jobs/i/out/` and the task log to `.../jobs/i/log.txt`,
   then exits with the task's status

The GitHub token comes from Secrets Manager (`gnomepy/gh-token`) and is never in the task's environment.

## Writing a job module

Any module runnable with `python -m` works. Read inputs from the paths your args name (`/work/in/<name>/...`),
from datasets (`DatasetStore().load(name, as_of=...)`) or artifacts (`resolve_artifact_path`), and write everything
to keep under `$OUTPUT_DIR`. Submit with `research jobs submit` from gnomepy-research.
