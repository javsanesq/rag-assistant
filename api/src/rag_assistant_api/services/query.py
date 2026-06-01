from __future__ import annotations

import re
import time

from rag_assistant_api.adapters.llm import LLMProvider
from rag_assistant_api.adapters.reranker import RerankDecision, RerankerProvider
from rag_assistant_api.adapters.vector_store import RetrievedChunk
from rag_assistant_api.core.config import Settings
from rag_assistant_api.domain.schemas import Citation, QueryRequest, QueryResponse
from rag_assistant_api.services.retrieval import RetrievalService


class QueryService:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_provider: LLMProvider,
        reranker_provider: RerankerProvider,
        settings: Settings,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.llm_provider = llm_provider
        self.reranker_provider = reranker_provider
        self.settings = settings

    def answer_question(self, request: QueryRequest) -> QueryResponse:
        started = time.perf_counter()
        retrieved = self.retrieval_service.retrieve(request)
        relevant, rejected = _filter_relevant_chunks(retrieved, self.settings, request.question)
        rerank_decision = None
        if relevant and (request.rerank or request.answerability_check):
            rerank_decision = self.reranker_provider.rerank(request.question, relevant)
            if request.answerability_check and not rerank_decision.answerable:
                return self._insufficient_evidence_response(
                    request,
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                    rejected=rejected,
                    rerank_decision=rerank_decision,
                    warning=rerank_decision.rationale,
                )
            if request.rerank:
                relevant = rerank_decision.selected_chunks
        if not relevant:
            return self._insufficient_evidence_response(
                request,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                rejected=rejected,
                rerank_decision=rerank_decision,
            )

        context_blocks = [
            (
                f"[{index + 1}] chunk_id={item.chunk_id} document_id={item.document_id} "
                f"title={item.title}\n{item.excerpt}"
            )
            for index, item in enumerate(relevant)
        ]
        raw_answer = self.llm_provider.answer(request.question, context_blocks)
        answer, grounding = _validate_or_repair_answer(raw_answer, {str(index + 1): item for index, item in enumerate(relevant)})
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        citations = [_to_citation(item) for item in relevant]
        trace = None
        if request.include_trace:
            trace = {
                "retrieval_mode": request.retrieval_mode,
                "alpha": request.alpha,
                "selected_chunks": [_trace_chunk(item, relevant=True) for item in relevant],
                "rejected_chunks": [_trace_chunk(item, relevant=False) for item in rejected],
                **_rerank_trace(rerank_decision, relevant),
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
            metrics={"latency_ms": elapsed_ms, "retrieved_count": len(citations), "rejected_count": len(rejected)},
            trace=trace,
            grounded=grounding["grounded"],
            used_citation_ids=grounding["used_citation_ids"],
            warnings=grounding["warnings"],
        )

    def _insufficient_evidence_response(
        self,
        request: QueryRequest,
        elapsed_ms: float,
        rejected: list[RetrievedChunk],
        rerank_decision: RerankDecision | None = None,
        warning: str | None = None,
    ) -> QueryResponse:
        trace = None
        if request.include_trace:
            trace = {
                "retrieval_mode": request.retrieval_mode,
                "alpha": request.alpha,
                "selected_chunks": [],
                "rejected_chunks": [_trace_chunk(item, relevant=False) for item in rejected],
                **_rerank_trace(rerank_decision, []),
            }
        warnings = [warning] if warning else ["Retrieved chunks did not meet the relevance threshold for this query."]
        return QueryResponse(
            answer="I do not have enough evidence in the indexed documents to answer that.",
            citations=[],
            applied_filters={
                "document_ids": request.document_ids,
                "category": request.category,
                "date_from": request.date_from.isoformat() if request.date_from else None,
                "date_to": request.date_to.isoformat() if request.date_to else None,
                "top_k": request.top_k or self.retrieval_service.default_top_k,
            },
            metrics={"latency_ms": elapsed_ms, "retrieved_count": 0, "rejected_count": len(rejected)},
            trace=trace,
            grounded=False,
            used_citation_ids=[],
            warnings=warnings,
        )


def _validate_or_repair_answer(answer: str, citation_map: dict[str, object]) -> tuple[str, dict]:
    if not citation_map:
        return _abstain("No citations were retrieved for this query.")

    cited_numbers = _extract_citation_numbers(answer)
    valid_numbers = sorted({number for number in cited_numbers if number in citation_map}, key=int)
    invalid_numbers = sorted({number for number in cited_numbers if number not in citation_map}, key=int)
    if invalid_numbers:
        invalid_markers = ", ".join(f"[{item}]" for item in invalid_numbers)
        return _abstain(f"Model answer used unsupported citation markers: {invalid_markers}.")

    if valid_numbers:
        return (
            answer.strip(),
            {
                "grounded": True,
                "used_citation_ids": [citation_map[number].chunk_id for number in valid_numbers],
                "warnings": [],
            },
        )

    return _abstain("Model answer did not cite retrieved context; abstained instead of returning an unsupported answer.")


def _abstain(warning: str) -> tuple[str, dict]:
    return (
        "I do not have enough evidence in the indexed documents to answer that.",
        {
            "grounded": False,
            "used_citation_ids": [],
            "warnings": [warning],
        },
    )


def _extract_citation_numbers(answer: str) -> list[str]:
    return re.findall(r"\[(\d+)\]", answer)


def _filter_relevant_chunks(chunks: list[RetrievedChunk], settings: Settings, question: str = "") -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    relevant = []
    rejected = []
    question_terms = _meaningful_terms(question)
    for chunk in chunks:
        if _is_relevant_chunk(chunk, settings, question_terms):
            relevant.append(chunk)
        else:
            rejected.append(chunk)
    return relevant, rejected


def _is_relevant_chunk(chunk: RetrievedChunk, settings: Settings, question_terms: set[str]) -> bool:
    meaningful_overlap = len(question_terms & _meaningful_terms(chunk.excerpt))
    has_meaningful_overlap = meaningful_overlap >= settings.relevance_min_meaningful_terms
    lexical_relevant = chunk.lexical_score >= settings.relevance_min_lexical_score
    dense_relevant = chunk.dense_score >= settings.relevance_min_dense_score
    final_relevant = chunk.final_score >= settings.relevance_min_final_score
    return dense_relevant or lexical_relevant or (has_meaningful_overlap and final_relevant)


def _to_citation(item: RetrievedChunk) -> Citation:
    return Citation(
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


def _trace_chunk(item: RetrievedChunk, relevant: bool) -> dict:
    return {
        "chunk_id": item.chunk_id,
        "document_id": item.document_id,
        "dense_score": item.dense_score,
        "lexical_score": item.lexical_score,
        "final_score": item.final_score,
        "relevant": relevant,
    }


def _rerank_trace(decision: RerankDecision | None, selected_chunks: list[RetrievedChunk]) -> dict:
    if not decision:
        return {
            "reranker_provider": None,
            "reranker_model": None,
            "answerable": None,
            "reranker_rationale": None,
            "candidate_chunk_ids": [chunk.chunk_id for chunk in selected_chunks],
            "selected_chunk_ids": [chunk.chunk_id for chunk in selected_chunks],
        }
    return {
        "reranker_provider": decision.provider,
        "reranker_model": decision.model,
        "answerable": decision.answerable,
        "reranker_rationale": decision.rationale,
        "candidate_chunk_ids": decision.candidate_chunk_ids,
        "selected_chunk_ids": decision.selected_chunk_ids,
    }


_STOPWORDS = {
    "about",
    "after",
    "and",
    "are",
    "back",
    "backs",
    "does",
    "for",
    "from",
    "has",
    "how",
    "into",
    "the",
    "this",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _meaningful_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]{3,}", text.lower())
        if token not in _STOPWORDS and not token.isdigit()
    }
