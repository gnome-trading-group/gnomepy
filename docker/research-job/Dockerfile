# syntax=docker/dockerfile:1.6
#
# Image that runs any gnomepy-research module at a pinned commit, as one task of a research job
# (gnome-controller /research-jobs). gnomepy and every gnomepy-research dependency family are baked in;
# the research repo itself is cloned at run time so one image serves any commit.
#
# Build context must be the parent directory containing both `gnomepy` and `gnomepy-research` checkouts:
#
#   ./gnomepy/docker/research-job/build.sh
#
# See gnomepy/docker/research-job/README.md.

FROM python:3.13-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1

# g++: pytensor (pymc) compiles its graph to C and falls back to slow pure-Python ops without a compiler.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# gnomepy from this working tree, and gnomepy-research's dependencies at its locked versions (constraints.txt).
# gnomepy-research itself is removed again: each task installs the commit it was submitted with.
COPY gnomepy /opt/gnomepy
COPY gnomepy-research/pyproject.toml gnomepy-research/constraints.txt gnomepy-research/README.md /opt/gnomepy-research/
RUN mkdir -p /opt/gnomepy-research/gnomepy_research && touch /opt/gnomepy-research/gnomepy_research/__init__.py && \
    EXTRAS=$(python3 -c "import tomllib; print(','.join(tomllib.load(open('/opt/gnomepy-research/pyproject.toml', 'rb'))['project']['optional-dependencies']))") && \
    pip install -c /opt/gnomepy-research/constraints.txt "/opt/gnomepy[strategy,live-data,backtest]" "/opt/gnomepy-research[${EXTRAS}]" && \
    pip uninstall -y gnomepy-research && \
    rm -rf /opt/gnomepy-research

COPY gnomepy/docker/research-job/run_task.py /usr/local/bin/run_task.py

WORKDIR /work
ENTRYPOINT ["python3", "/usr/local/bin/run_task.py"]
