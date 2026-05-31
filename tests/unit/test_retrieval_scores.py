from rag_assistant_api.adapters.vector_store import _lexical_score, _tokenize
from rag_assistant_api.adapters.lexical_store import SQLLexicalStore
from rag_assistant_api.adapters.vector_store import RetrievedChunk
from rag_assistant_api.core.db import build_engine, build_session_factory
from rag_assistant_api.domain.models import ChunkRecord
from rag_assistant_api.domain.schemas import QueryRequest
from rag_assistant_api.services.retrieval import RetrievalService


def test_lexical_score_counts_query_term_overlap():
    query_terms = _tokenize("annual refund policy")
    chunk_terms = _tokenize("annual plans have a refund window")
    assert _lexical_score(query_terms, chunk_terms) == 2 / 3


def test_hybrid_retrieval_can_return_lexical_candidate_when_dense_misses():
    service = RetrievalService(
        vector_store=FakeVectorStore(),
        embedding_provider=FakeEmbeddingProvider(),
        default_top_k=1,
        lexical_store=FakeLexicalStore(),
    )

    results = service.retrieve(
        QueryRequest(
            question="annual refund policy",
            top_k=1,
            retrieval_mode="hybrid",
            alpha=0.25,
        )
    )

    assert [item.document_id for item in results] == ["lexical-document"]
    assert results[0].lexical_score == 1.0


def test_sqlite_fts_bm25_lexical_store_ranks_exact_terms(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'fts.db'}")
    ChunkRecord.metadata.create_all(engine)
    session_factory = build_session_factory(engine)
    with session_factory() as session:
        session.add_all(
            [
                ChunkRecord(
                    chunk_id="exact",
                    document_id="security-runbook",
                    title="Security Runbook",
                    source_uri="security.md",
                    source_type="markdown",
                    category="security",
                    chunk_index=0,
                    chunk_text="The emergency rotation keyword is KESTREL-42-ZETA.",
                    lexical_terms_json='["emergency", "rotation", "keyword", "kestrel", "zeta"]',
                ),
                ChunkRecord(
                    chunk_id="weak",
                    document_id="general",
                    title="General",
                    source_uri="general.md",
                    source_type="markdown",
                    category="security",
                    chunk_index=0,
                    chunk_text="The weekly security meeting reviews access requests.",
                    lexical_terms_json='["weekly", "security", "meeting", "access"]',
                ),
            ]
        )
        session.commit()

    results = SQLLexicalStore(session_factory).search("What is the KESTREL-42-ZETA keyword?", top_k=2)

    assert results[0].chunk_id == "exact"
    assert results[0].document_id == "security-runbook"


class FakeEmbeddingProvider:
    def embed_query(self, text: str) -> list[float]:
        return [0.0]


class FakeVectorStore:
    def search(self, **kwargs) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="dense",
                document_id="dense-document",
                title="Dense Document",
                source_uri="dense.md",
                category=None,
                document_date=None,
                excerpt="Semantically nearby but lexically weak.",
                score=0.8,
                dense_score=0.8,
                lexical_score=0.0,
                final_score=0.8,
                chunk_index=0,
            )
        ]


class FakeLexicalStore:
    def search(self, **kwargs) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="lexical",
                document_id="lexical-document",
                title="Lexical Document",
                source_uri="lexical.md",
                category=None,
                document_date=None,
                excerpt="Annual refund policy exact match.",
                score=1.0,
                dense_score=0.0,
                lexical_score=1.0,
                final_score=1.0,
                chunk_index=0,
            )
        ]
