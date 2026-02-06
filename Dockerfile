ARG PYTHON_VERSION=3.12-slim

FROM python:$PYTHON_VERSION AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock .

RUN uv export --locked >requirements.txt

FROM python:$PYTHON_VERSION

WORKDIR /app

COPY --from=builder /app/requirements.txt .
COPY model_filter.py config.yaml.default .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

WORKDIR /config

VOLUME ["/config"]
EXPOSE 8080/tcp

ENV LLM_MF_HOST="0.0.0.0"
CMD ["/usr/local/bin/python", "/app/model_filter.py"]
