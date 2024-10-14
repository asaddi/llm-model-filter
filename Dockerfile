FROM python:3.12-slim

WORKDIR /app

COPY model_filter.py config.yaml.default requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

WORKDIR /config

VOLUME ["/config"]
EXPOSE 8080/tcp

ENV LLM_MF_HOST="0.0.0.0"
CMD ["/usr/local/bin/python", "/app/model_filter.py"]
