FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY api/pyproject.toml /workspace/api/pyproject.toml
COPY api/src /workspace/api/src
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -e /workspace/api

COPY evals /workspace/evals
COPY samples /workspace/samples

RUN mkdir -p /workspace/data

WORKDIR /workspace/api

CMD ["uvicorn", "rag_assistant_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
