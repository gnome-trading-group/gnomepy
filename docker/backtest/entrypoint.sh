#!/usr/bin/env bash
#
# Clone gnomepy-research at $RESEARCH_COMMIT, install it, and run a backtest.
#
# Required env:
#   RESEARCH_COMMIT  git SHA / ref of gnomepy-research to check out
#   GH_TOKEN         GitHub token with read access to gnomepy-research
#   BACKTEST_CONFIG  s3:// URI or path (inside container) to YAML backtest config
#
# AWS_* env vars must be passed for S3 access (config download, market data, results).
#
# Any extra args are forwarded to `gnomepy backtest`.

set -euo pipefail

: "${RESEARCH_COMMIT:?RESEARCH_COMMIT is required}"
: "${GH_TOKEN:?GH_TOKEN is required}"
: "${BACKTEST_CONFIG:?BACKTEST_CONFIG is required}"

LOCAL_CONFIG=/work/config.yaml

# If config is an S3 URI, download it via boto3.
if [[ "$BACKTEST_CONFIG" == s3://* ]]; then
  echo "entrypoint: downloading config from $BACKTEST_CONFIG"
  python3 - <<PY
import boto3, sys
bucket, key = "$BACKTEST_CONFIG"[5:].split("/", 1)
boto3.client("s3").download_file(bucket, key, "$LOCAL_CONFIG")
PY
  BACKTEST_CONFIG=$LOCAL_CONFIG
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
gnomepy backtest --config "$BACKTEST_CONFIG" "$@"
