#!/usr/bin/env bash
#
# Clone gnomepy-research at $RESEARCH_COMMIT, install it, and run a backtest.
#
# Required env:
#   RUN_ID               backtest run identifier (hex timestamp)
#   S3_BUCKET            S3 bucket for configs and results (e.g. gnome-research-prod)
#   RESEARCH_COMMIT      git SHA / ref of gnomepy-research to check out
#   AWS_DEFAULT_REGION   AWS region (injected by CDK job definition)
#
# GH_TOKEN is fetched from AWS Secrets Manager (secret name: gnomepy/gh-token).

set -euo pipefail

: "${RUN_ID:?RUN_ID is required}"
: "${S3_BUCKET:?S3_BUCKET is required}"
: "${RESEARCH_COMMIT:?RESEARCH_COMMIT is required}"

ARRAY_INDEX="${JOB_INDEX:-0}"
CONFIG_S3_KEY="backtests/${RUN_ID}/jobs/${ARRAY_INDEX}/config.yaml"
OUTPUT_PREFIX="s3://${S3_BUCKET}/backtests/${RUN_ID}/jobs/${ARRAY_INDEX}"
LOCAL_CONFIG=/work/config.yaml

: "${AWS_DEFAULT_REGION:?AWS_DEFAULT_REGION is required}"

echo "entrypoint: run_id=${RUN_ID} array_index=${ARRAY_INDEX} region=${AWS_DEFAULT_REGION}"

echo "entrypoint: fetching gh-token from Secrets Manager"
GH_TOKEN=$(python3 - <<'PY'
import boto3, json
client = boto3.client("secretsmanager")
secret = client.get_secret_value(SecretId="gnomepy/gh-token")
val = secret["SecretString"]
try:
    print(json.loads(val)["token"])
except (json.JSONDecodeError, KeyError):
    print(val.strip())
PY
)

echo "entrypoint: downloading config from s3://${S3_BUCKET}/${CONFIG_S3_KEY}"
python3 - <<PY
import boto3
boto3.client("s3").download_file("${S3_BUCKET}", "${CONFIG_S3_KEY}", "${LOCAL_CONFIG}")
PY

REPO_DIR=/opt/gnomepy-research

echo "entrypoint: cloning gnomepy-research..."
git clone --filter=blob:none --quiet \
  "https://x-access-token:${GH_TOKEN}@github.com/gnome-trading-group/gnomepy-research.git" \
  "$REPO_DIR"

echo "entrypoint: checking out ${RESEARCH_COMMIT}"
git -C "$REPO_DIR" checkout --quiet "$RESEARCH_COMMIT"
git -C "$REPO_DIR" --no-pager log -1 --oneline

# --no-deps so the pinned gnomepy in this image is not replaced.
echo "entrypoint: installing gnomepy-research"
pip install --quiet --no-deps "$REPO_DIR"

unset GH_TOKEN

echo "entrypoint: running backtest (output -> ${OUTPUT_PREFIX})"
gnomepy backtest run \
  --config "${LOCAL_CONFIG}" \
  --output "${OUTPUT_PREFIX}"
