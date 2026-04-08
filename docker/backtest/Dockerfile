# syntax=docker/dockerfile:1.6
#
# Image that runs a `gnomepy backtest` against a pinned commit of the
# private gnomepy-research repo.
#
# Build context must be the parent directory containing both `gnomepy`
# and `gnome-backtest` checkouts, e.g.:
#
#   docker build \
#     -f gnomepy/docker/backtest/Dockerfile \
#     -t gnomepy-backtest \
#     /path/to/src/gnome
#
# See gnomepy/docker/backtest/README.md for run instructions.

# ---------- Stage 1: build the gnome-backtest uber JAR ----------
FROM maven:3.9-eclipse-temurin-21 AS jar-builder

# GitHub Packages credentials needed to resolve the gnome-parent POM.
# Pass GITHUB_ACTOR as a build-arg and GITHUB_TOKEN as a BuildKit secret:
#   docker build \
#     --build-arg GITHUB_ACTOR=$GITHUB_ACTOR \
#     --secret id=github_token,env=GITHUB_TOKEN \
#     -f gnomepy/docker/backtest/Dockerfile -t gnomepy-backtest .
ARG GITHUB_ACTOR
# Optional: pin gnome-backtest to a remote git ref (tag/branch/SHA).
# When empty, the local sibling checkout (COPY'd below) is used instead.
ARG GNOME_BACKTEST_REF=

WORKDIR /build
COPY gnome-backtest /build/gnome-backtest-local

RUN --mount=type=cache,target=/root/.m2 \
    --mount=type=secret,id=github_token,target=/run/secrets/github_token \
    set -eu; \
    export GITHUB_TOKEN="$(cat /run/secrets/github_token)"; \
    if [ -n "$GNOME_BACKTEST_REF" ]; then \
      echo ">>> cloning gnome-backtest @ $GNOME_BACKTEST_REF"; \
      git clone --filter=blob:none \
        "https://x-access-token:${GITHUB_TOKEN}@github.com/gnome-trading-group/gnome-backtest.git" \
        /build/gnome-backtest; \
      git -C /build/gnome-backtest checkout "$GNOME_BACKTEST_REF"; \
    else \
      echo ">>> using local gnome-backtest sibling checkout"; \
      cp -a /build/gnome-backtest-local /build/gnome-backtest; \
    fi; \
    cd /build/gnome-backtest && \
    mvn -q -DskipTests --settings settings.xml package


# ---------- Stage 2: runtime ----------
FROM python:3.13-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    GNOME_ROOT=/opt/gnome \
    JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# JARs from stage 1. _classpath.py scans $GNOME_ROOT/gnome-backtest/target
# for *-all.jar, so preserve that layout.
RUN mkdir -p /opt/gnome/gnome-backtest/target
COPY --from=jar-builder /build/gnome-backtest/target/*-all.jar /opt/gnome/gnome-backtest/target/

# Install gnomepy itself (pinned to the working tree we are building from).
COPY gnomepy /opt/gnomepy
RUN pip install /opt/gnomepy

COPY gnomepy/docker/backtest/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /work
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
