from __future__ import annotations

import re
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
        citation_map = {str(index + 1): item for index, item in enumerate(retrieved)}
        context_blocks = [
            (
                f"[{index + 1}] chunk_id={item.chunk_id} document_id={item.document_id} "
                f"title={item.title}\n{item.excerpt}"
            )
            for index, item in enumerate(retrieved)
        ]
        raw_answer = self.llm_provider.answer(request.question, context_blocks)
        answer, grounding = _validate_or_repair_answer(raw_answer, citation_map)
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
            grounded=grounding["grounded"],
            used_citation_ids=grounding["used_citation_ids"],
            warnings=grounding["warnings"],
        )


def _validate_or_repair_answer(answer: str, citation_map: dict[str, object]) -> tuple[str, dict]:
    warnings: list[str] = []
    if not citation_map:
        return (
            "I do not have enough evidence in the indexed documents to answer that.",
            {
                "grounded": False,
                "used_citation_ids": [],
                "warnings": ["No citations were retrieved for this query."],
            },
        )

    cited_numbers = _extract_citation_numbers(answer)
    valid_numbers = sorted({number for number in cited_numbers if number in citation_map}, key=int)
    invalid_numbers = sorted({number for number in cited_numbers if number not in citation_map}, key=int)
    if invalid_numbers:
        warnings.append(f"Removed unsupported citation markers: {', '.join(f'[{item}]' for item in invalid_numbers)}.")
        answer = _strip_invalid_citations(answer, set(invalid_numbers))

    if valid_numbers:
        return (
            answer.strip(),
            {
                "grounded": not invalid_numbers,
                "used_citation_ids": [citation_map[number].chunk_id for number in valid_numbers],
                "warnings": warnings,
            },
        )

    fallback_number = "1"
    fallback = citation_map[fallback_number]
    warnings.append("Model answer did not cite retrieved context; returned a citation-grounded fallback.")
    fallback_answer = f"Based on the retrieved context: {fallback.excerpt[:450].strip()} [{fallback_number}]"
    return (
        fallback_answer,
        {
            "grounded": True,
            "used_citation_ids": [fallback.chunk_id],
            "warnings": warnings,
        },
    )


def _extract_citation_numbers(answer: str) -> list[str]:
    return re.findall(r"\[(\d+)\]", answer)


def _strip_invalid_citations(answer: str, invalid_numbers: set[str]) -> str:
    for number in invalid_numbers:
        answer = re.sub(rf"\[{re.escape(number)}\]", "", answer)
    return re.sub(r"\s+", " ", answer).strip()
