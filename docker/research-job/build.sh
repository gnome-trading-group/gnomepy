#!/usr/bin/env bash
#
# Build the gnomepy-research-job image.
#
# Must be run from anywhere inside a parent directory holding both `gnomepy` and `gnomepy-research`
# checkouts (this script finds the context from its own location).
#
# Optional env:
#   IMAGE_TAG   image tag to build (default: gnomepy-research-job:latest)

set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-gnomepy-research-job:latest}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [[ ! -d "$CONTEXT_DIR/gnomepy" || ! -d "$CONTEXT_DIR/gnomepy-research" ]]; then
  echo "build.sh: expected sibling checkouts at $CONTEXT_DIR/{gnomepy,gnomepy-research}" >&2
  exit 1
fi

echo "build.sh: building $IMAGE_TAG from $CONTEXT_DIR"
DOCKER_BUILDKIT=1 docker build \
  -f "$CONTEXT_DIR/gnomepy/docker/research-job/Dockerfile" \
  -t "$IMAGE_TAG" \
  "$CONTEXT_DIR" \
  "$@"

echo "build.sh: built $IMAGE_TAG"
