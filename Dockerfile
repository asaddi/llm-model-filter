ARG PYTHON_VERSION=3.12-slim

FROM python:$PYTHON_VERSION AS build

COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /bin/

WORKDIR /app

COPY pyproject.toml uv.lock .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --exact --locked --link-mode=copy --compile-bytecode

FROM python:$PYTHON_VERSION

WORKDIR /app

COPY --from=build /app/.venv .venv
COPY model_filter.py config.yaml.default .

WORKDIR /config

VOLUME ["/config"]
EXPOSE 8080/tcp

ENV LLM_MF_HOST="0.0.0.0"
CMD ["/app/.venv/bin/python", "/app/model_filter.py"]
