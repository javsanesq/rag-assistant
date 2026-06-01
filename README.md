# RAG Assistant

[![CI](https://github.com/javsanesq/rag-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/javsanesq/rag-assistant/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production-grade RAG assistant built with FastAPI, Qdrant, durable SQL-backed jobs, offline evaluation, visible citations, and a polished operations UI.

## What it demonstrates

- Multi-source ingestion: PDF, DOCX, Markdown, URL, and sitemap/list-based URL batches
- Configurable chunking and metadata normalization
- Embeddings plus Qdrant vector retrieval with document, date, and category filters
- Hybrid retrieval that fuses dense Qdrant candidates with an SQL-backed lexical chunk index
- Citation-rich answers with applied-filter telemetry
- Citation-grounded answer validation with used chunk IDs and grounding warnings
- Retrieval relevance gating that abstains when retrieved chunks are too weak to support the question
- Optional reranking and answerability checks with `none`, `mock`, and OpenAI providers
- Offline evaluation with document-level metrics, chunk-level metrics, answer-content checks, tag summaries, and a faithfulness rubric
- Structured logging, durable worker jobs, tests, Docker, and a strong portfolio UI

## Stack

- API: FastAPI + SQLAlchemy + Qdrant client
- Storage: Qdrant for vectors, SQLite by default for metadata/jobs, Postgres via `DATABASE_URL`, and Alembic for versioned schema migrations
- Parsing: `pypdf`, `python-docx`, `beautifulsoup4`, `PyYAML`
- Embeddings: mock for smoke tests, OpenAI for production, optional sentence-transformers for local models
- LLM: mock, OpenAI, or Ollama
- Reranker: none, mock heuristic, or OpenAI answerability reranker
- UI: static HTML/CSS/JS behind nginx

## Quickstart

```bash
git clone https://github.com/javsanesq/rag-assistant.git
cd rag-assistant
cp .env.example .env
docker compose up --build
```

Open:

- UI: [http://localhost:3000](http://localhost:3000)
- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Qdrant: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

`docker compose` starts four services: API, worker, Qdrant, and UI. The API enqueues ingestion/evaluation jobs; the worker claims and executes them.
The API and worker apply Alembic migrations at startup, so a fresh `DATABASE_URL` is initialized automatically.
The checked-in `.env.example` uses `API_AUTH_TOKEN=change-me-dev-token`; paste that token into the UI token field for the local demo, and replace it before any real deployment.

## Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
make test
make api
```

Install local sentence-transformers support only when you need offline embedding models:

```bash
cd api
pip install -e '.[local-embeddings]'
```

Run the worker locally in a second terminal:

```bash
source .venv/bin/activate
cd api && PYTHONPATH=src python -m rag_assistant_api.worker
```

## Configuration

This repository intentionally commits `.env.example` and ignores real `.env` files. The example file documents the runtime contract with safe placeholder values; secrets such as `OPENAI_API_KEY` must live only in your local environment, deployment secret manager, or untracked `.env`.

Important defaults:

- `EMBED_PROVIDER=mock` and `LLM_PROVIDER=mock` keep Docker smoke tests deterministic and usable without paid API keys.
- Set `LLM_PROVIDER=openai` and `OPENAI_API_KEY=...` for hosted answer generation.
- Set `RERANKER_PROVIDER=openai` and `OPENAI_API_KEY=...` for hosted reranking and answerability checks.
- Set `DATABASE_URL=postgresql+psycopg://...` when deploying against Postgres instead of local SQLite.
- Relevance thresholds are configurable through `RELEVANCE_MIN_*` environment variables.
- `/api/v1/*` routes fail closed when `API_AUTH_TOKEN` is empty. Set a strong token before exposing the API; clients may send it as `x-api-key` or `Authorization: Bearer ...`.
- `POST /api/v1/query` and `POST /api/v1/documents/urls` have simple per-client rate limits controlled by `API_QUERY_RATE_LIMIT_PER_MINUTE`, `API_URL_INGEST_RATE_LIMIT_PER_MINUTE`, and `API_RATE_LIMIT_WINDOW_SECONDS`.
- When `APP_ENV=production`, mock providers are rejected and `API_AUTH_TOKEN` is required.
- Docker Compose binds demo ports to `127.0.0.1` by default. Put Qdrant on a private network and expose only the UI/API through a reverse proxy for production.

## Database migrations

The SQL schema is managed with Alembic. Runtime startup automatically upgrades the configured database to the latest migration, which keeps local Docker, VPS, and Postgres-backed deployments consistent.

Manual migration commands:

```bash
make db-upgrade
make db-revision m="describe change"
```

SQLite is the default local database. Postgres works through `DATABASE_URL`, for example:

```bash
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/rag_assistant make db-upgrade
```

## Core flows

### File ingestion

`POST /api/v1/documents/files`

- Accepts one or more files
- Optional `metadata_json`
- Optional `chunker_type`, `chunk_size`, `chunk_overlap`
- Queues an ingestion job and persists document metadata
- Stores accepted uploads under `data/uploads/`
- Rejects unsupported extensions, empty files, and oversized files before enqueueing

### URL ingestion

`POST /api/v1/documents/urls`

- Accepts a single URL, a list of URLs, or a sitemap URL
- Uses the same normalization and chunking pipeline as file ingestion
- Blocks private/local URL targets by default and caps response/sitemap size

### Querying

`POST /api/v1/query`

- Accepts `question`
- Optional `document_ids`, `category`, `date_from`, `date_to`
- Returns `answer`, `citations`, `applied_filters`, and timing metadata
- Supports `retrieval_mode`, `alpha`, and `include_trace` for hybrid dense/lexical retrieval diagnostics
- Supports optional `rerank` and `answerability_check` for reranker-backed candidate selection and no-answer decisions
- Hybrid mode independently retrieves lexical candidates, so exact-match terms can rescue chunks missed by dense search
- Validates that generated answers cite retrieved context; uncited or invalidly cited answers abstain instead of being forced into a fallback
- Applies configurable relevance thresholds before answer generation. If retrieved chunks are too weak, the API returns insufficient evidence instead of forcing a cited answer.

### Evaluation

`POST /api/v1/evals/runs`

- Runs against checked-in datasets under `evals/datasets`
- Computes retrieval metrics and a faithfulness rubric
- Stores per-example rationale and aggregate summaries
- Reports `precision@k`, `hit_rate@k`, `recall@k`, `mrr`, faithfulness, abstention accuracy, unsupported-answer rate, citation relevance, and per-filter summaries
- Supports richer benchmark rows with `expected_chunk_ids`, `expected_answer_contains`, `tags`, `difficulty`, and `answerability_reason`
- Reports chunk-level `precision@k`, hit rate, recall, MRR, answer-content match rate, and per-tag summaries

### Jobs

`GET /api/v1/jobs/{job_id}`

- Returns status, progress, attempts, errors, payload, and result summary

`POST /api/v1/jobs/{job_id}/retry`

- Requeues failed jobs without creating a duplicate job record

## Smoke checks

```bash
docker compose config
docker compose up --build
curl -f http://localhost:8000/health/live
curl -f http://localhost:8000/health/ready
node --check ui/app.js
```

Run the full Docker RAG workflow smoke after the stack is running:

```bash
make smoke-e2e
```

The E2E smoke uploads the checked-in sample knowledge base, waits for the worker to ingest it, verifies document inventory, runs a grounded filtered query with citations, starts an offline evaluation run, waits for completion, and checks the nginx UI endpoint. To let the script start the stack itself:

```bash
START_STACK=1 make smoke-e2e
```

If you changed `API_AUTH_TOKEN`, pass the same value to the smoke script:

```bash
API_AUTH_TOKEN=your-token START_STACK=1 make smoke-e2e
```

## Benchmark

Run the OpenAI-backed benchmark to compare dense retrieval, SQLite FTS5 BM25 hybrid retrieval, production relevance thresholds, and OpenAI reranking:

```bash
OPENAI_API_KEY=... make benchmark
```

The benchmark indexes `samples/benchmark`, evaluates `evals/datasets/benchmark_eval.jsonl`, writes the committed report to `docs/benchmark-report.md`, and stores raw local results under `output/evals/benchmark-results.json`.
The current report includes exact-match, semantic, lexical, filter, date-conflict, prompt-injection, multi-hop, near-miss, and no-answer cases.

## Repository layout

```text
rag-assistant/
├── api/                    # FastAPI package
├── docs/                   # Architecture and implementation notes
├── evals/datasets/         # Offline evaluation sets
├── samples/knowledge/      # Example knowledge base docs
├── tests/                  # Unit and integration tests
└── ui/                     # Static operations UI
```

## Project Notes

The API and worker live in the same Python package because they share domain models, database access, providers, and service-layer code. Docker Compose starts them as separate processes: the API handles HTTP requests, while the worker claims durable jobs and executes ingestion/evaluation outside the request path.
