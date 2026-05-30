from rag_assistant_api.adapters.vector_store import RetrievedChunk
from rag_assistant_api.services.query import _validate_or_repair_answer


def test_grounded_answer_accepts_valid_citation_marker():
    answer, grounding = _validate_or_repair_answer("Refunds are available for 30 days [1].", {"1": _chunk("chunk-a")})

    assert answer == "Refunds are available for 30 days [1]."
    assert grounding["grounded"] is True
    assert grounding["used_citation_ids"] == ["chunk-a"]
    assert grounding["warnings"] == []


def test_uncited_answer_is_repaired_with_cited_fallback():
    answer, grounding = _validate_or_repair_answer("Refunds are available for 30 days.", {"1": _chunk("chunk-a")})

    assert answer.endswith("[1]")
    assert "Refunds are available within 30 calendar days" in answer
    assert grounding["grounded"] is True
    assert grounding["used_citation_ids"] == ["chunk-a"]
    assert grounding["warnings"]


def test_no_retrieved_citations_returns_insufficient_evidence():
    answer, grounding = _validate_or_repair_answer("Anything", {})

    assert answer == "I do not have enough evidence in the indexed documents to answer that."
    assert grounding["grounded"] is False
    assert grounding["used_citation_ids"] == []
    assert grounding["warnings"] == ["No citations were retrieved for this query."]


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="company-handbook",
        title="Company Handbook",
        source_uri="company-handbook.md",
        category="policy",
        document_date=None,
        excerpt="Refunds are available within 30 calendar days of purchase.",
        score=1.0,
        dense_score=1.0,
        lexical_score=1.0,
        final_score=1.0,
        chunk_index=0,
    )
