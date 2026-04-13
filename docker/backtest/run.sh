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
#   OUTPUT_DIR       host dir mounted at /work/out (default: ./out)
#   JOB_ID           job id (default: generated UUIDv7)
#   BACKTEST_BUCKET  s3 bucket for default output (default: gnome-research)
#   SKIP_AWS=1       do not resolve/forward AWS credentials
#
# Extra args are forwarded to `gnomepy backtest`. `--output` may be either
# a local path or an s3:// URI; s3 paths are uploaded after the run.
#
# Examples:
#   ./run.sh --output /work/out/results.json
#   ./run.sh --output s3://my-bucket/backtests/run-42.json
#   BACKTEST_CONFIG=s3://my-bucket/configs/momentum.yaml ./run.sh \
#       --output s3://my-bucket/backtests/momentum.json

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

IMAGE_TAG="${IMAGE_TAG:-gnomepy-backtest:latest}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/out}"
mkdir -p "$OUTPUT_DIR"

# UUIDv7 (time-ordered). Falls back to python if uuidgen not present.
gen_uuid7() {
  python3 - <<'PY'
import os, secrets, time
ms = int(time.time() * 1000) & 0xFFFFFFFFFFFF
ra = secrets.randbits(12)
rb = secrets.randbits(62)
n = (ms << 80) | (0x7 << 76) | (ra << 64) | (0b10 << 62) | rb
h = f"{n:032x}"
print(f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}")
PY
}
: "${JOB_ID:=$(gen_uuid7)}"
: "${BACKTEST_BUCKET:=gnome-research}"
echo "run.sh: job_id=$JOB_ID"

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

# ---------- Resolve config (local or s3://) -------------------------------
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

# ---------- Default --output to s3://$BACKTEST_BUCKET/backtests/$JOB_ID/ --
HAS_OUTPUT=0
for a in "$@"; do
  case "$a" in --output|-o|--output=*|-o=*) HAS_OUTPUT=1; break;; esac
done
if (( ! HAS_OUTPUT )); then
  set -- "$@" --output "s3://$BACKTEST_BUCKET/backtests/$JOB_ID/"
fi

# Always forward the job id to the CLI so manifest matches the s3 prefix.
set -- "$@" --job-id "$JOB_ID"

# ---------- Rewrite --output if it's s3:// --------------------------------
# Walk the forwarded args, replacing any s3:// --output with a container path
# and remembering the URI for upload after the run.
OUTPUT_S3=""
FORWARDED_ARGS=()
i=0
args=("$@")
while (( i < ${#args[@]} )); do
  a="${args[$i]}"
  case "$a" in
    --output|-o)
      val="${args[$((i+1))]:-}"
      if [[ -z "$val" ]]; then
        echo "run.sh: --output requires a value" >&2
        exit 1
      fi
      if [[ "$val" == s3://* ]]; then
        OUTPUT_S3="$val"
        FORWARDED_ARGS+=("$a" "/work/out/$(basename "${val%/}")")
      else
        FORWARDED_ARGS+=("$a" "$val")
      fi
      i=$((i+2))
      ;;
    --output=*|-o=*)
      val="${a#*=}"
      if [[ "$val" == s3://* ]]; then
        OUTPUT_S3="$val"
        FORWARDED_ARGS+=("--output" "/work/out/$(basename "${val%/}")")
      else
        FORWARDED_ARGS+=("$a")
      fi
      i=$((i+1))
      ;;
    *)
      FORWARDED_ARGS+=("$a")
      i=$((i+1))
      ;;
  esac
done

# ---------- Run the container ---------------------------------------------
echo "run.sh: $IMAGE_TAG @ commit $RESEARCH_COMMIT"
set +e
docker run --rm \
  -e RESEARCH_COMMIT \
  -e GH_TOKEN \
  -e BACKTEST_CONFIG=/work/backtest.yaml \
  "${AWS_ARGS[@]}" \
  -v "$CONFIG_ABS:/work/backtest.yaml:ro" \
  -v "$OUTPUT_DIR:/work/out" \
  "$IMAGE_TAG" \
  "${FORWARDED_ARGS[@]}"
rc=$?
set -e

# ---------- Upload result if requested ------------------------------------
if [[ -n "$OUTPUT_S3" && $rc -eq 0 ]]; then
  local_result="$OUTPUT_DIR/$(basename "$OUTPUT_S3")"
  dest="${OUTPUT_S3%/}"
  if [[ -d "$local_result" ]]; then
    echo "run.sh: uploading $local_result/ -> $dest/"
    aws s3 cp "$local_result" "$dest/" --recursive --quiet
  elif [[ -f "$local_result" ]]; then
    echo "run.sh: uploading $local_result -> $OUTPUT_S3"
    aws s3 cp "$local_result" "$OUTPUT_S3" --quiet
  else
    echo "run.sh: warning: expected result not found at $local_result" >&2
  fi
fi

exit "$rc"
