FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/*

COPY .. /app

RUN python -m pip install --upgrade pip && \
    pip install .

ENV PYTHONPATH=/app
ENV PORT=7860
ENV ENABLE_WEB_INTERFACE=true

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f "http://localhost:${PORT}/health" || exit 1

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
