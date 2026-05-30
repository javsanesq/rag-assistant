from __future__ import annotations

import time

from rag_assistant_api.adapters.llm import LLMProvider
from rag_assistant_api.domain.schemas import Citation, QueryRequest, QueryResponse
from rag_assistant_api.services.retrieval import RetrievalService


class QueryService:
    def __init__(self, retrieval_service: RetrievalService, llm_provider: LLMProvider) -> None:
        self.retrieval_service = retrieval_service
        self.llm_provider = llm_provider

    def answer_question(self, request: QueryRequest) -> QueryResponse:
        started = time.perf_counter()
        retrieved = self.retrieval_service.retrieve(request)
        context_blocks = [f"[{index + 1}] {item.excerpt}" for index, item in enumerate(retrieved)]
        answer = self.llm_provider.answer(request.question, context_blocks)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        citations = [
            Citation(
                document_id=item.document_id,
                title=item.title,
                source_uri=item.source_uri,
                category=item.category,
                document_date=item.document_date,
                excerpt=item.excerpt[:360],
                score=item.score,
                dense_score=item.dense_score,
                lexical_score=item.lexical_score,
                final_score=item.final_score,
                chunk_id=item.chunk_id,
                chunk_index=item.chunk_index,
            )
            for item in retrieved
        ]
        trace = None
        if request.include_trace:
            trace = {
                "retrieval_mode": request.retrieval_mode,
                "alpha": request.alpha,
                "selected_chunks": [
                    {
                        "chunk_id": item.chunk_id,
                        "document_id": item.document_id,
                        "dense_score": item.dense_score,
                        "lexical_score": item.lexical_score,
                        "final_score": item.final_score,
                    }
                    for item in retrieved
                ],
            }
        return QueryResponse(
            answer=answer,
            citations=citations,
            applied_filters={
                "document_ids": request.document_ids,
                "category": request.category,
                "date_from": request.date_from.isoformat() if request.date_from else None,
                "date_to": request.date_to.isoformat() if request.date_to else None,
                "top_k": request.top_k or self.retrieval_service.default_top_k,
            },
            metrics={"latency_ms": elapsed_ms, "retrieved_count": len(citations)},
            trace=trace,
        )
