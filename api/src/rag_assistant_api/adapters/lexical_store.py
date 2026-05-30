from __future__ import annotations

import json
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from rag_assistant_api.adapters.vector_store import RetrievedChunk, _lexical_score, _tokenize
from rag_assistant_api.domain.models import ChunkRecord


class SQLLexicalStore:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def search(
        self,
        query_text: str,
        top_k: int,
        document_ids: list[str] | None = None,
        category: str | None = None,
        date_from_timestamp: int | None = None,
        date_to_timestamp: int | None = None,
    ) -> list[RetrievedChunk]:
        query_terms = _tokenize(query_text)
        if not query_terms:
            return []

        with self.session_factory() as session:
            stmt = select(ChunkRecord)
            if document_ids:
                stmt = stmt.where(ChunkRecord.document_id.in_(document_ids))
            if category:
                stmt = stmt.where(ChunkRecord.category == category)
            if date_from_timestamp is not None:
                stmt = stmt.where(ChunkRecord.document_timestamp >= date_from_timestamp)
            if date_to_timestamp is not None:
                stmt = stmt.where(ChunkRecord.document_timestamp <= date_to_timestamp)
            records = session.scalars(stmt).all()

        candidates = []
        for record in records:
            terms = json.loads(record.lexical_terms_json or "[]")
            lexical_score = _lexical_score(query_terms, terms)
            if lexical_score <= 0:
                continue
            candidates.append(_to_retrieved_chunk(record, lexical_score))
        return sorted(candidates, key=lambda item: item.lexical_score, reverse=True)[:top_k]


def _to_retrieved_chunk(record: ChunkRecord, lexical_score: float) -> RetrievedChunk:
    document_date: date | None = record.document_date
    return RetrievedChunk(
        chunk_id=record.chunk_id,
        document_id=record.document_id,
        title=record.title,
        source_uri=record.source_uri,
        category=record.category,
        document_date=document_date,
        excerpt=record.chunk_text,
        score=lexical_score,
        dense_score=0.0,
        lexical_score=lexical_score,
        final_score=lexical_score,
        chunk_index=record.chunk_index,
    )
