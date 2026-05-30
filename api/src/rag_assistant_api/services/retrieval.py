from __future__ import annotations

from datetime import date

from rag_assistant_api.adapters.embeddings import EmbeddingProvider
from rag_assistant_api.adapters.vector_store import QdrantVectorStore, RetrievedChunk
from rag_assistant_api.domain.schemas import QueryRequest
from rag_assistant_api.services.metadata import to_timestamp


class RetrievalService:
    def __init__(self, vector_store: QdrantVectorStore, embedding_provider: EmbeddingProvider, default_top_k: int) -> None:
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.default_top_k = default_top_k

    def retrieve(self, request: QueryRequest) -> list[RetrievedChunk]:
        query_vector = self.embedding_provider.embed_query(request.question)
        return self.vector_store.search(
            query_vector=query_vector,
            top_k=request.top_k or self.default_top_k,
            document_ids=request.document_ids,
            category=request.category,
            date_from_timestamp=to_timestamp(request.date_from),
            date_to_timestamp=to_timestamp(request.date_to),
            query_text=request.question,
            retrieval_mode=request.retrieval_mode,
            alpha=request.alpha,
        )
