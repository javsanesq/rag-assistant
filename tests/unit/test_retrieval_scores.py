from rag_assistant_api.adapters.vector_store import _lexical_score, _tokenize
from rag_assistant_api.adapters.lexical_store import SQLLexicalStore
from rag_assistant_api.adapters.vector_store import RetrievedChunk
from rag_assistant_api.core.db import build_engine, build_session_factory
from rag_assistant_api.domain.models import ChunkRecord
from rag_assistant_api.domain.schemas import QueryRequest
from rag_assistant_api.services.retrieval import RetrievalService, _fuse_candidates


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


def test_hybrid_fusion_uses_rank_not_raw_score_scale():
    dense_first = _retrieved_chunk("dense-first", dense_score=0.95, lexical_score=0.0)
    lexical_first = _retrieved_chunk("lexical-first", dense_score=0.05, lexical_score=1.0)

    results = _fuse_candidates(
        dense_hits=[dense_first, lexical_first],
        lexical_hits=[lexical_first],
        top_k=2,
        alpha=0.5,
    )

    assert results[0].chunk_id == "lexical-first"
    assert results[0].final_score > results[1].final_score


def test_sqlite_fts_bm25_lexical_store_ranks_exact_terms(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'fts.db'}")
    ChunkRecord.metadata.create_all(engine)
    session_factory = build_session_factory(engine)
    with session_factory() as session:
        session.add_all(
            [
                _chunk_record(
                    chunk_id="exact",
                    document_id="security-runbook",
                    title="Security Runbook",
                    source_uri="security.md",
                    category="security",
                    chunk_text="The emergency rotation keyword is KESTREL-42-ZETA.",
                    lexical_terms_json='["emergency", "rotation", "keyword", "kestrel", "zeta"]',
                ),
                _chunk_record(
                    chunk_id="weak",
                    document_id="general",
                    title="General",
                    source_uri="general.md",
                    category="security",
                    chunk_text="The weekly security meeting reviews access requests.",
                    lexical_terms_json='["weekly", "security", "meeting", "access"]',
                ),
            ]
        )
        session.commit()

    results = SQLLexicalStore(session_factory).search("What is the KESTREL-42-ZETA keyword?", top_k=2)

    assert results[0].chunk_id == "exact"
    assert results[0].document_id == "security-runbook"


def test_sqlite_fts_bm25_filters_before_ranking(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'filtered.db'}")
    ChunkRecord.metadata.create_all(engine)
    session_factory = build_session_factory(engine)
    with session_factory() as session:
        session.add_all(
            [
                _chunk_record(
                    chunk_id=f"global-{index}",
                    document_id=f"global-{index}",
                    title="Global",
                    source_uri=f"global-{index}.md",
                    category="general",
                    chunk_text=f"Global KESTREL policy distractor {index}.",
                )
                for index in range(40)
            ]
        )
        session.add(
            _chunk_record(
                chunk_id="target",
                document_id="target-doc",
                title="Target",
                source_uri="target.md",
                category="target",
                chunk_text="Target category KESTREL answer.",
            )
        )
        session.commit()

    results = SQLLexicalStore(session_factory).search("KESTREL", top_k=1, category="target")

    assert [item.chunk_id for item in results] == ["target"]


def test_sqlite_fts_bm25_index_tracks_equal_count_replacements(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'stale.db'}")
    ChunkRecord.metadata.create_all(engine)
    session_factory = build_session_factory(engine)
    with session_factory() as session:
        session.add_all(
            [
                _chunk_record(
                    chunk_id="old",
                    document_id="old-doc",
                    title="Old",
                    source_uri="old.md",
                    category="policy",
                    chunk_text="The KESTREL procedure used to live here.",
                ),
                _chunk_record(
                    chunk_id="new",
                    document_id="new-doc",
                    title="New",
                    source_uri="new.md",
                    category="policy",
                    chunk_text="This chunk does not contain the keyword yet.",
                ),
            ]
        )
        session.commit()

    store = SQLLexicalStore(session_factory)
    assert store.search("KESTREL", top_k=1)[0].chunk_id == "old"

    with session_factory() as session:
        old = session.get(ChunkRecord, "old")
        new = session.get(ChunkRecord, "new")
        old.chunk_text = "The procedure moved away from this chunk."
        new.chunk_text = "The KESTREL procedure now lives here."
        session.add_all([old, new])
        session.commit()

    assert store.search("KESTREL", top_k=1)[0].chunk_id == "new"


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


def _retrieved_chunk(chunk_id: str, dense_score: float, lexical_score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=chunk_id,
        title=chunk_id,
        source_uri=f"{chunk_id}.md",
        category=None,
        document_date=None,
        excerpt=chunk_id,
        score=dense_score,
        dense_score=dense_score,
        lexical_score=lexical_score,
        final_score=dense_score,
        chunk_index=0,
    )


def _chunk_record(
    chunk_id: str,
    document_id: str,
    title: str,
    source_uri: str,
    category: str,
    chunk_text: str,
    source_type: str = "markdown",
    chunk_index: int = 0,
    lexical_terms_json: str = "[]",
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        title=title,
        source_uri=source_uri,
        source_type=source_type,
        category=category,
        chunk_index=chunk_index,
        chunk_text=chunk_text,
        lexical_terms_json=lexical_terms_json,
    )
