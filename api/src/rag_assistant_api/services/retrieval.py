from __future__ import annotations

from rag_assistant_api.adapters.embeddings import EmbeddingProvider
from rag_assistant_api.adapters.lexical_store import SQLLexicalStore
from rag_assistant_api.adapters.vector_store import QdrantVectorStore, RetrievedChunk
from rag_assistant_api.domain.schemas import QueryRequest
from rag_assistant_api.services.metadata import to_timestamp


class RetrievalService:
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embedding_provider: EmbeddingProvider,
        default_top_k: int,
        lexical_store: SQLLexicalStore | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.default_top_k = default_top_k
        self.lexical_store = lexical_store

    def retrieve(self, request: QueryRequest) -> list[RetrievedChunk]:
        query_vector = self.embedding_provider.embed_query(request.question)
        top_k = request.top_k or self.default_top_k
        date_from_timestamp = to_timestamp(request.date_from)
        date_to_timestamp = to_timestamp(request.date_to)
        candidate_limit = top_k * 4 if request.retrieval_mode == "hybrid" else top_k
        dense_hits = self.vector_store.search(
            query_vector=query_vector,
            top_k=candidate_limit,
            document_ids=request.document_ids,
            category=request.category,
            date_from_timestamp=date_from_timestamp,
            date_to_timestamp=date_to_timestamp,
            query_text=request.question,
            retrieval_mode="dense",
            alpha=1.0,
        )
        if request.retrieval_mode == "dense" or not self.lexical_store:
            return dense_hits[:top_k]

        lexical_hits = self.lexical_store.search(
            query_text=request.question,
            top_k=candidate_limit,
            document_ids=request.document_ids,
            category=request.category,
            date_from_timestamp=date_from_timestamp,
            date_to_timestamp=date_to_timestamp,
        )
        return _fuse_candidates(dense_hits, lexical_hits, top_k=top_k, alpha=request.alpha)


def _fuse_candidates(
    dense_hits: list[RetrievedChunk],
    lexical_hits: list[RetrievedChunk],
    top_k: int,
    alpha: float,
) -> list[RetrievedChunk]:
    merged: dict[str, RetrievedChunk] = {}
    for hit in [*dense_hits, *lexical_hits]:
        existing = merged.get(hit.chunk_id)
        if existing is None:
            merged[hit.chunk_id] = hit
            continue
        existing.dense_score = max(existing.dense_score, hit.dense_score)
        existing.lexical_score = max(existing.lexical_score, hit.lexical_score)

    for hit in merged.values():
        dense_for_final = _normalize_dense_score(hit.dense_score)
        hit.final_score = alpha * dense_for_final + (1 - alpha) * hit.lexical_score
        hit.score = hit.final_score
    return sorted(merged.values(), key=lambda item: item.final_score, reverse=True)[:top_k]


def _normalize_dense_score(score: float) -> float:
    return max(0.0, min(1.0, score))
