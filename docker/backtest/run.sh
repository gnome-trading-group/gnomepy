#!/usr/bin/env bash
#
# Run the gnomepy-backtest image against a pinned commit of gnomepy-research.
#
# Required env:
#   RESEARCH_COMMIT  git SHA / ref of gnomepy-research
#   GH_TOKEN         GitHub token with read access to gnomepy-research
#                    (falls back to GITHUB_TOKEN)
#
# Optional env:
#   BACKTEST_CONFIG  host path OR s3:// URI to a YAML config
#                    (default: example-backtest.yaml next to this script)
#   IMAGE_TAG        image to run (default: gnomepy-backtest:latest)
#   OUTPUT_DIR       host dir mounted at /work/out for local --output (default: ./out)
#   BACKTEST_BUCKET  s3 bucket for default --s3-bucket (default: gnome-research)
#   SKIP_AWS=1       do not resolve/forward AWS credentials
#
# Extra args are forwarded unchanged to `gnomepy backtest`.
#
# Examples:
#   ./run.sh
#   ./run.sh --s3-bucket my-bucket
#   BACKTEST_CONFIG=s3://my-bucket/configs/momentum.yaml ./run.sh

set -euo pipefail

: "${RESEARCH_COMMIT:?RESEARCH_COMMIT is required}"
: "${GH_TOKEN:=${GITHUB_TOKEN:-}}"
: "${GH_TOKEN:?GH_TOKEN (or GITHUB_TOKEN) is required}"
export GH_TOKEN

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Allow --config / -c as a shorthand for BACKTEST_CONFIG. Strip it from $@
# so we don't double-pass it to `gnomepy backtest`.
PASSTHROUGH=()
i=0
all=("$@")
while (( i < ${#all[@]} )); do
  a="${all[$i]}"
  case "$a" in
    --config|-c)
      BACKTEST_CONFIG="${all[$((i+1))]:-}"
      [[ -z "$BACKTEST_CONFIG" ]] && { echo "run.sh: --config requires a value" >&2; exit 1; }
      i=$((i+2))
      ;;
    --config=*|-c=*)
      BACKTEST_CONFIG="${a#*=}"
      i=$((i+1))
      ;;
    *)
      PASSTHROUGH+=("$a")
      i=$((i+1))
      ;;
  esac
done
set -- "${PASSTHROUGH[@]}"

: "${BACKTEST_CONFIG:=$SCRIPT_DIR/example-backtest.yaml}"
: "${BACKTEST_BUCKET:=gnome-research}"

IMAGE_TAG="${IMAGE_TAG:-gnomepy-backtest:latest}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/out}"
mkdir -p "$OUTPUT_DIR"

TMPDIR_RUN="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_RUN"' EXIT

# ---------- AWS credentials ------------------------------------------------
# Resolve via host AWS CLI so SSO / profile / role-assumption all "just work".
AWS_ARGS=()
if [[ -z "${SKIP_AWS:-}" ]]; then
  if command -v aws >/dev/null 2>&1; then
    if creds_json="$(aws configure export-credentials --format process 2>/dev/null)"; then
      AWS_ACCESS_KEY_ID="$(printf '%s' "$creds_json"     | sed -n 's/.*"AccessKeyId"[^"]*"\([^"]*\)".*/\1/p')"
      AWS_SECRET_ACCESS_KEY="$(printf '%s' "$creds_json" | sed -n 's/.*"SecretAccessKey"[^"]*"\([^"]*\)".*/\1/p')"
      AWS_SESSION_TOKEN="$(printf '%s' "$creds_json"     | sed -n 's/.*"SessionToken"[^"]*"\([^"]*\)".*/\1/p')"
      export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
    else
      echo "run.sh: warning: 'aws configure export-credentials' failed; using existing AWS_* env vars" >&2
    fi
  fi
  : "${AWS_REGION:=${AWS_DEFAULT_REGION:-us-east-1}}"
  export AWS_REGION
  if [[ -z "${AWS_ACCESS_KEY_ID:-}" || -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
    echo "run.sh: ERROR: no AWS credentials resolved." >&2
    echo "  - install/configure the AWS CLI (aws sso login --profile <p>), or" >&2
    echo "  - export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, or" >&2
    echo "  - re-run with SKIP_AWS=1 if the backtest does not need S3" >&2
    exit 1
  fi
  echo "run.sh: AWS creds resolved (key=${AWS_ACCESS_KEY_ID:0:4}..., region=$AWS_REGION)"
  AWS_ARGS+=(-e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_REGION)
  [[ -n "${AWS_SESSION_TOKEN:-}" ]] && AWS_ARGS+=(-e AWS_SESSION_TOKEN)
fi

# ---------- Resolve config (local or s3://) --------------------------------
if [[ "$BACKTEST_CONFIG" == s3://* ]]; then
  command -v aws >/dev/null 2>&1 || { echo "run.sh: aws CLI required for s3:// config" >&2; exit 1; }
  CONFIG_LOCAL="$TMPDIR_RUN/backtest.yaml"
  echo "run.sh: downloading config from $BACKTEST_CONFIG"
  aws s3 cp "$BACKTEST_CONFIG" "$CONFIG_LOCAL" --quiet
else
  CONFIG_LOCAL="$BACKTEST_CONFIG"
  if [[ ! -f "$CONFIG_LOCAL" ]]; then
    echo "run.sh: BACKTEST_CONFIG not found: $CONFIG_LOCAL" >&2
    exit 1
  fi
fi
CONFIG_ABS="$(cd "$(dirname "$CONFIG_LOCAL")" && pwd)/$(basename "$CONFIG_LOCAL")"

# ---------- Default --s3-bucket if no output specified --------------------
HAS_OUTPUT=0
for a in "$@"; do
  case "$a" in --output|-o|--output=*|-o=*|--s3-bucket|--s3-bucket=*) HAS_OUTPUT=1; break;; esac
done
if (( ! HAS_OUTPUT )); then
  set -- "$@" --s3-bucket "$BACKTEST_BUCKET"
fi

# ---------- Run the container ---------------------------------------------
echo "run.sh: $IMAGE_TAG @ commit $RESEARCH_COMMIT"
docker run --rm \
  -e RESEARCH_COMMIT \
  -e GH_TOKEN \
  -e BACKTEST_CONFIG=/work/backtest.yaml \
  "${AWS_ARGS[@]}" \
  -v "$CONFIG_ABS:/work/backtest.yaml:ro" \
  -v "$OUTPUT_DIR:/work/out" \
  "$IMAGE_TAG" \
  "$@"
