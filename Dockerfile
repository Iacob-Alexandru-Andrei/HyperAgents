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

WORKDIR /hyperagents

COPY requirements-core.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements-core.txt

COPY . .

CMD ["tail", "-f", "/dev/null"]
