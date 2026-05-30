from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from rag_assistant_api.adapters.embeddings import build_embedding_provider
from rag_assistant_api.adapters.llm import build_llm_provider
from rag_assistant_api.adapters.vector_store import QdrantVectorStore
from rag_assistant_api.api import documents, evals, health, jobs, query
from rag_assistant_api.core.config import Settings
from rag_assistant_api.core.db import build_engine, build_session_factory, init_db
from rag_assistant_api.core.logging import bind_request_id, configure_logging
from rag_assistant_api.services.documents import DocumentService
from rag_assistant_api.services.evaluation import EvaluationService
from rag_assistant_api.services.jobs import JobService
from rag_assistant_api.services.query import QueryService
from rag_assistant_api.services.retrieval import RetrievalService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    configure_logging(settings.log_level)
    engine = build_engine(settings.effective_database_url)
    session_factory = build_session_factory(engine)
    init_db(engine)
    embedding_provider = build_embedding_provider(settings)
    llm_provider = build_llm_provider(settings)
    vector_store = QdrantVectorStore(settings, embedding_provider.dimensions)
    job_service = JobService(session_factory)
    document_service = DocumentService(session_factory, vector_store, embedding_provider, job_service, settings)
    retrieval_service = RetrievalService(vector_store, embedding_provider, settings.top_k)
    query_service = QueryService(retrieval_service, llm_provider)
    evaluation_service = EvaluationService(settings, query_service, job_service, llm_provider)

    app.state.settings = settings
    app.state.session_factory = session_factory
    app.state.vector_store = vector_store
    app.state.job_service = job_service
    app.state.document_service = document_service
    app.state.query_service = query_service
    app.state.evaluation_service = evaluation_service
    yield
    vector_store.close()
    engine.dispose()


app = FastAPI(title="RAG Assistant API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=Settings().api_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    bind_request_id(request.headers.get("x-request-id"))
    response = await call_next(request)
    return response


app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(evals.router)
app.include_router(jobs.router)
