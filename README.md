# RAG Assistant

Production-grade RAG assistant built with FastAPI, Qdrant, SQLite-backed job metadata, offline evaluation, visible citations, and a polished operations UI.

## What it demonstrates

- Multi-source ingestion: PDF, DOCX, Markdown, URL, and sitemap/list-based URL batches
- Configurable chunking and metadata normalization
- Embeddings plus Qdrant vector retrieval with document, date, and category filters
- Citation-rich answers with applied-filter telemetry
- Offline evaluation with `precision@k`, `hit_rate@k`, and a faithfulness rubric
- Structured logging, background jobs, tests, Docker, and a strong portfolio UI

## Stack

- API: FastAPI + SQLAlchemy + Qdrant client
- Storage: Qdrant for vectors, SQLite for documents and jobs
- Parsing: `pypdf`, `python-docx`, `beautifulsoup4`, `PyYAML`
- Embeddings: sentence-transformers or OpenAI
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

## Local development

```bash
cd /Users/javiersanchezesquivel/Desktop/Proyectos/rag-assistant
python3 -m venv .venv
source .venv/bin/activate
make install
make test
make api
```

## Core flows

### File ingestion

`POST /api/v1/documents/files`

- Accepts one or more files
- Optional `metadata_json`
- Optional `chunker_type`, `chunk_size`, `chunk_overlap`
- Queues an ingestion job and persists document metadata

### URL ingestion

`POST /api/v1/documents/urls`

- Accepts a single URL, a list of URLs, or a sitemap URL
- Uses the same normalization and chunking pipeline as file ingestion

### Querying

`POST /api/v1/query`

- Accepts `question`
- Optional `document_ids`, `category`, `date_from`, `date_to`
- Returns `answer`, `citations`, `applied_filters`, and timing metadata

### Evaluation

`POST /api/v1/evals/runs`

- Runs against checked-in datasets under `evals/datasets`
- Computes retrieval metrics and a faithfulness rubric
- Stores per-example rationale and aggregate summaries

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
