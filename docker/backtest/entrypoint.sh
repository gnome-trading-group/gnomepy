#!/usr/bin/env bash
#
# Clone gnomepy-research at $RESEARCH_COMMIT, install it, and run a backtest.
#
# Required env:
#   RESEARCH_COMMIT  git SHA / ref of gnomepy-research to check out
#   GH_TOKEN         GitHub token with read access to gnomepy-research
#   BACKTEST_CONFIG  path (inside container) to YAML backtest config
#
# AWS_* env vars should also be passed if the backtest reads from S3.
#
# Any extra args are forwarded to `gnomepy backtest`, e.g.:
#   docker run ... gnomepy-research-backtest --output /work/out/results.json

set -euo pipefail

: "${RESEARCH_COMMIT:?RESEARCH_COMMIT is required}"
: "${GH_TOKEN:?GH_TOKEN is required}"
: "${BACKTEST_CONFIG:?BACKTEST_CONFIG is required (path inside container)}"

# If config is an S3 URI, download it to a local path.
if [[ "$BACKTEST_CONFIG" == s3://* ]]; then
  echo "entrypoint: downloading config from $BACKTEST_CONFIG"
  aws s3 cp "$BACKTEST_CONFIG" /work/config.yaml --quiet
  BACKTEST_CONFIG=/work/config.yaml
fi

if [[ ! -f "$BACKTEST_CONFIG" ]]; then
  echo "entrypoint: BACKTEST_CONFIG not found at $BACKTEST_CONFIG" >&2
  exit 1
fi

REPO_DIR=/opt/gnomepy-research

echo "entrypoint: cloning gnomepy-research..."
git clone --filter=blob:none --quiet \
  "https://x-access-token:${GH_TOKEN}@github.com/gnome-trading-group/gnomepy-research.git" \
  "$REPO_DIR"

echo "entrypoint: checking out $RESEARCH_COMMIT"
git -C "$REPO_DIR" checkout --quiet "$RESEARCH_COMMIT"
git -C "$REPO_DIR" --no-pager log -1 --oneline

# --no-deps so the pinned gnomepy in this image is not replaced.
echo "entrypoint: installing gnomepy-research"
pip install --quiet --no-deps "$REPO_DIR"

# Scrub the token from the env before running the backtest.
unset GH_TOKEN

echo "entrypoint: running gnomepy backtest"
exec gnomepy backtest --config "$BACKTEST_CONFIG" "$@"
