from rag_assistant_api.adapters.vector_store import RetrievedChunk
from rag_assistant_api.core.config import Settings
from rag_assistant_api.services.query import _filter_relevant_chunks, _validate_or_repair_answer


def test_grounded_answer_accepts_valid_citation_marker():
    answer, grounding = _validate_or_repair_answer("Refunds are available for 30 days [1].", {"1": _chunk("chunk-a")})

    assert answer == "Refunds are available for 30 days [1]."
    assert grounding["grounded"] is True
    assert grounding["used_citation_ids"] == ["chunk-a"]
    assert grounding["warnings"] == []


def test_uncited_answer_abstains_instead_of_using_fallback_chunk():
    answer, grounding = _validate_or_repair_answer("Refunds are available for 30 days.", {"1": _chunk("chunk-a")})

    assert answer == "I do not have enough evidence in the indexed documents to answer that."
    assert grounding["grounded"] is False
    assert grounding["used_citation_ids"] == []
    assert grounding["warnings"] == [
        "Model answer did not cite retrieved context; abstained instead of returning an unsupported answer."
    ]


def test_invalid_citation_marker_abstains_even_with_valid_citation_present():
    answer, grounding = _validate_or_repair_answer(
        "Refunds are available for 30 days [1], and expedited refunds take two days [9].",
        {"1": _chunk("chunk-a")},
    )

    assert answer == "I do not have enough evidence in the indexed documents to answer that."
    assert grounding["grounded"] is False
    assert grounding["used_citation_ids"] == []
    assert grounding["warnings"] == ["Model answer used unsupported citation markers: [9]."]


def test_only_invalid_citation_marker_does_not_fallback_to_first_chunk():
    answer, grounding = _validate_or_repair_answer("Refunds are available for 30 days [7].", {"1": _chunk("chunk-a")})

    assert answer == "I do not have enough evidence in the indexed documents to answer that."
    assert grounding["grounded"] is False
    assert grounding["used_citation_ids"] == []
    assert grounding["warnings"] == ["Model answer used unsupported citation markers: [7]."]


def test_no_retrieved_citations_returns_insufficient_evidence():
    answer, grounding = _validate_or_repair_answer("Anything", {})

    assert answer == "I do not have enough evidence in the indexed documents to answer that."
    assert grounding["grounded"] is False
    assert grounding["used_citation_ids"] == []
    assert grounding["warnings"] == ["No citations were retrieved for this query."]


def test_relevance_gate_rejects_dense_only_noise():
    settings = Settings()

    relevant, rejected = _filter_relevant_chunks(
        [_chunk("dense-noise", dense_score=0.4, lexical_score=0.0, final_score=0.4)],
        settings,
        "What is the Zurich office phone number?",
    )

    assert relevant == []
    assert rejected[0].chunk_id == "dense-noise"


def test_relevance_gate_accepts_lexical_match():
    settings = Settings()

    relevant, rejected = _filter_relevant_chunks(
        [_chunk("lexical-hit", dense_score=0.0, lexical_score=0.3, final_score=0.075)],
        settings,
        "What is the refund calendar window?",
    )

    assert relevant[0].chunk_id == "lexical-hit"
    assert rejected == []


def _chunk(chunk_id: str, dense_score: float = 1.0, lexical_score: float = 1.0, final_score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="company-handbook",
        title="Company Handbook",
        source_uri="company-handbook.md",
        category="policy",
        document_date=None,
        excerpt="Refunds are available within 30 calendar days of purchase.",
        score=final_score,
        dense_score=dense_score,
        lexical_score=lexical_score,
        final_score=final_score,
        chunk_index=0,
    )
