from rag_assistant_api.adapters.vector_store import _lexical_score, _tokenize
from rag_assistant_api.adapters.vector_store import RetrievedChunk
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
