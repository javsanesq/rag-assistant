from rag_assistant_api.adapters.reranker import MockRerankerProvider, NoopRerankerProvider, _safe_json
from rag_assistant_api.adapters.vector_store import RetrievedChunk


def test_noop_reranker_preserves_candidate_order():
    chunks = [_chunk("a"), _chunk("b")]

    decision = NoopRerankerProvider().rerank("What is indexed?", chunks)

    assert decision.answerable is True
    assert decision.selected_chunk_ids == ["a", "b"]
    assert decision.provider == "none"


def test_mock_reranker_rejects_unrelated_near_miss_context():
    decision = MockRerankerProvider().rerank(
        "What is the Zurich disaster recovery phone number?",
        [_chunk("near-miss", excerpt="The Zurich office does not host disaster recovery infrastructure.", dense_score=0.2, lexical_score=0.1)],
    )

    assert decision.answerable is False
    assert decision.selected_chunks == []


def test_mock_reranker_selects_direct_overlap():
    decision = MockRerankerProvider().rerank(
        "What is the annual refund window?",
        [_chunk("refund", excerpt="Annual plan refunds are available within 30 calendar days.", lexical_score=0.5)],
    )

    assert decision.answerable is True
    assert decision.selected_chunk_ids == ["refund"]


def test_safe_json_extracts_json_from_wrapped_model_output():
    parsed = _safe_json('```json\n{"answerable": true, "chunk_ids": ["a"], "rationale": "ok"}\n```')

    assert parsed["answerable"] is True
    assert parsed["chunk_ids"] == ["a"]


def _chunk(chunk_id: str, excerpt: str = "Annual refund policy.", dense_score: float = 0.8, lexical_score: float = 0.2) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=chunk_id,
        title=chunk_id,
        source_uri=f"{chunk_id}.md",
        category=None,
        document_date=None,
        excerpt=excerpt,
        score=dense_score,
        dense_score=dense_score,
        lexical_score=lexical_score,
        final_score=max(dense_score, lexical_score),
        chunk_index=0,
    )
