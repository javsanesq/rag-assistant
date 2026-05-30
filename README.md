# RAG Assistant

Production-grade RAG assistant built with FastAPI, Qdrant, durable SQL-backed jobs, offline evaluation, visible citations, and a polished operations UI.

## What it demonstrates

- Multi-source ingestion: PDF, DOCX, Markdown, URL, and sitemap/list-based URL batches
- Configurable chunking and metadata normalization
- Embeddings plus Qdrant vector retrieval with document, date, and category filters
- Hybrid retrieval that fuses dense Qdrant candidates with an SQL-backed lexical chunk index
- Citation-rich answers with applied-filter telemetry
- Citation-grounded answer validation with used chunk IDs and grounding warnings
- Retrieval relevance gating that abstains when retrieved chunks are too weak to support the question
- Offline evaluation with `precision@k`, `hit_rate@k`, and a faithfulness rubric
- Structured logging, durable worker jobs, tests, Docker, and a strong portfolio UI

## Stack

- API: FastAPI + SQLAlchemy + Qdrant client
- Storage: Qdrant for vectors, SQLite by default for metadata/jobs, Postgres via `DATABASE_URL`, and Alembic for versioned schema migrations
- Parsing: `pypdf`, `python-docx`, `beautifulsoup4`, `PyYAML`
- Embeddings: mock for smoke tests, OpenAI for production, optional sentence-transformers for local models
- LLM: mock, OpenAI, or Ollama
- UI: static HTML/CSS/JS behind nginx

## Quickstart

```bash
cd /Users/javiersanchezesquivel/Desktop/Proyectos/rag-assistant
cp .env.example .env
docker compose up --build
```

Open:

- UI: [http://localhost:3000](http://localhost:3000)
- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Qdrant: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

`docker compose` starts four services: API, worker, Qdrant, and UI. The API enqueues ingestion/evaluation jobs; the worker claims and executes them.
The API and worker apply Alembic migrations at startup, so a fresh `DATABASE_URL` is initialized automatically.

## Local development

```bash
cd /Users/javiersanchezesquivel/Desktop/Proyectos/rag-assistant
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
cd /Users/javiersanchezesquivel/Desktop/Proyectos/rag-assistant
source .venv/bin/activate
cd api && PYTHONPATH=src python -m rag_assistant_api.worker
```

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
- Hybrid mode independently retrieves lexical candidates, so exact-match terms can rescue chunks missed by dense search
- Validates that generated answers cite retrieved context; uncited answers are repaired with a citation-grounded fallback
- Applies configurable relevance thresholds before answer generation. If retrieved chunks are too weak, the API returns insufficient evidence instead of forcing a cited answer.

### Evaluation

`POST /api/v1/evals/runs`

- Runs against checked-in datasets under `evals/datasets`
- Computes retrieval metrics and a faithfulness rubric
- Stores per-example rationale and aggregate summaries
- Reports `precision@k`, `hit_rate@k`, `recall@k`, `mrr`, faithfulness, abstention accuracy, unsupported-answer rate, citation relevance, and per-filter summaries

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

## Repository layout

```text
rag-assistant/
├── api/                    # FastAPI package
├── docs/reference/         # Recovered reference notes from the earlier prototype
├── evals/datasets/         # Offline evaluation sets
├── samples/knowledge/      # Example knowledge base docs
├── tests/                  # Unit and integration tests
├── ui/                     # Static operations UI
└── worker/                 # CLI entrypoints and background-job notes
```

## Reference context

The earlier `rag-system` project in `/Users/javiersanchezesquivel/Desktop/Proyectos/rag-system` is retained as reference-only. This repository rebuilds the system around explicit service boundaries rather than a framework-led prototype.
