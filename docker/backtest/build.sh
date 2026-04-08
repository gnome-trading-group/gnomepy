#!/usr/bin/env bash
#
# Build the gnomepy-backtest image.
#
# Must be run from the parent directory containing both `gnomepy` and
# `gnome-backtest` checkouts (this script auto-cd's there).
#
# Required env:
#   GITHUB_ACTOR  GitHub username
#   GITHUB_TOKEN  classic PAT with read:packages (for GitHub Packages Maven)
#
# Optional env:
#   IMAGE_TAG           image tag to build (default: gnomepy-backtest:latest)
#   GNOME_BACKTEST_REF  git ref of gnome-backtest to clone & build instead of
#                       the local sibling checkout (e.g. v1.2.3, main, <sha>)

set -euo pipefail

: "${GITHUB_ACTOR:?GITHUB_ACTOR is required}"
: "${GITHUB_TOKEN:?GITHUB_TOKEN is required (classic PAT with read:packages)}"

IMAGE_TAG="${IMAGE_TAG:-gnomepy-backtest:latest}"

# cd to the parent of gnomepy/ (build context root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [[ ! -d "$CONTEXT_DIR/gnomepy" || ! -d "$CONTEXT_DIR/gnome-backtest" ]]; then
  echo "build.sh: expected sibling checkouts at $CONTEXT_DIR/{gnomepy,gnome-backtest}" >&2
  exit 1
fi

TOKEN_FILE="$(mktemp)"
trap 'rm -f "$TOKEN_FILE"' EXIT
printf '%s' "$GITHUB_TOKEN" >"$TOKEN_FILE"

if [[ -n "${GNOME_BACKTEST_REF:-}" ]]; then
  echo "build.sh: building $IMAGE_TAG from $CONTEXT_DIR (gnome-backtest @ $GNOME_BACKTEST_REF)"
else
  echo "build.sh: building $IMAGE_TAG from $CONTEXT_DIR (gnome-backtest from local sibling)"
fi

DOCKER_BUILDKIT=1 docker build \
  --build-arg GITHUB_ACTOR="$GITHUB_ACTOR" \
  --build-arg GNOME_BACKTEST_REF="${GNOME_BACKTEST_REF:-}" \
  --secret id=github_token,src="$TOKEN_FILE" \
  -f "$CONTEXT_DIR/gnomepy/docker/backtest/Dockerfile" \
  -t "$IMAGE_TAG" \
  "$CONTEXT_DIR" \
  "$@"

echo "build.sh: built $IMAGE_TAG"
