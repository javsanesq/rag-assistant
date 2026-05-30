from __future__ import annotations

from types import SimpleNamespace

from rag_assistant_api.adapters.embeddings import build_embedding_provider
from rag_assistant_api.adapters.lexical_store import SQLLexicalStore
from rag_assistant_api.adapters.llm import build_llm_provider
from rag_assistant_api.adapters.vector_store import QdrantVectorStore
from rag_assistant_api.core.config import Settings
from rag_assistant_api.core.db import build_engine, build_session_factory, init_db
from rag_assistant_api.core.logging import configure_logging
from rag_assistant_api.services.documents import DocumentService
from rag_assistant_api.services.evaluation import EvaluationService
from rag_assistant_api.services.jobs import JobService
from rag_assistant_api.services.query import QueryService
from rag_assistant_api.services.retrieval import RetrievalService


def build_runtime() -> SimpleNamespace:
    settings = Settings()
    configure_logging(settings.log_level)
    engine = build_engine(settings.effective_database_url)
    session_factory = build_session_factory(engine)
    init_db(engine)
    embedding_provider = build_embedding_provider(settings)
    llm_provider = build_llm_provider(settings)
    vector_store = QdrantVectorStore(settings, embedding_provider.dimensions)
    lexical_store = SQLLexicalStore(session_factory)
    job_service = JobService(session_factory, lease_seconds=settings.job_lease_seconds)
    document_service = DocumentService(session_factory, vector_store, embedding_provider, job_service, settings)
    retrieval_service = RetrievalService(vector_store, embedding_provider, settings.top_k, lexical_store)
    query_service = QueryService(retrieval_service, llm_provider)
    evaluation_service = EvaluationService(settings, query_service, job_service, llm_provider)
    return SimpleNamespace(
        settings=settings,
        engine=engine,
        session_factory=session_factory,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        vector_store=vector_store,
        lexical_store=lexical_store,
        job_service=job_service,
        document_service=document_service,
        retrieval_service=retrieval_service,
        query_service=query_service,
        evaluation_service=evaluation_service,
    )


def close_runtime(runtime: SimpleNamespace) -> None:
    runtime.vector_store.close()
    runtime.engine.dispose()
