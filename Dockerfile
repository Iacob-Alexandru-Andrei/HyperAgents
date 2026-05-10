FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    patch \
    && rm -rf /var/lib/apt/lists/*

# F2f: trust all directories for git inside this sandbox image. The
# /hyperagents and /REPO_NAME bind-mounts come from the host with the host
# user's UID, while the container process runs as root; without this,
# git rejects every operation with "dubious ownership".
RUN git config --global --add safe.directory '*'

WORKDIR /hyperagents

COPY requirements-core.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements-core.txt

COPY . .

CMD ["tail", "-f", "/dev/null"]
