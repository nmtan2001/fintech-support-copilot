FROM python:3.11-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39

ARG APP_VERSION=dev
ARG APP_COMMIT=unknown

LABEL org.opencontainers.image.title="Fintech Support Copilot" \
    org.opencontainers.image.revision=${APP_COMMIT} \
    org.opencontainers.image.version=${APP_VERSION} \
    org.opencontainers.image.source="https://github.com/nmtan2001/fintech-support-copilot"

ENV APP_VERSION=${APP_VERSION} \
    APP_COMMIT=${APP_COMMIT} \
    HF_HOME=/tmp/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    DEMO_DATABASE_PATH=/tmp/state/demo.sqlite

WORKDIR /app

COPY requirements-api.lock ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --timeout 180 --retries 3 --require-hashes -r requirements-api.lock

COPY pyproject.toml README.md api.py VERSION model-manifest.json ./
COPY fintech_support/ ./fintech_support/
COPY evaluate_smoke.py evaluate_retrieval.py analyze_errors.py ./
COPY data/faq/ ./data/faq/
COPY data/transactions/ ./data/transactions/
COPY data/evaluation/ ./data/evaluation/
COPY scripts/ ./scripts/

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /tmp/huggingface /tmp/state \
    && chown -R app:app /app /tmp/huggingface /tmp/state

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
