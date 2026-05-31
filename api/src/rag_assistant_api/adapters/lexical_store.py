from __future__ import annotations

import json
from datetime import date

from sqlalchemy import text, select
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
            if _sqlite_fts_available(session):
                records = _search_sqlite_fts(
                    session,
                    query_terms,
                    top_k,
                    document_ids,
                    category,
                    date_from_timestamp,
                    date_to_timestamp,
                )
                if records:
                    return records
            records = _filtered_chunk_records(session, document_ids, category, date_from_timestamp, date_to_timestamp)

        candidates = []
        for record in records:
            terms = json.loads(record.lexical_terms_json or "[]")
            lexical_score = _lexical_score(query_terms, terms)
            if lexical_score <= 0:
                continue
            candidates.append(_to_retrieved_chunk(record, lexical_score))
        return sorted(candidates, key=lambda item: item.lexical_score, reverse=True)[:top_k]


def _sqlite_fts_available(session: Session) -> bool:
    if session.bind is None or session.bind.dialect.name != "sqlite":
        return False
    try:
        session.execute(
            text(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts "
                "USING fts5(chunk_id UNINDEXED, chunk_text, tokenize='porter unicode61')"
            )
        )
        indexed = session.execute(text("SELECT COUNT(*) FROM chunks_fts")).scalar_one()
        chunk_count = session.execute(text("SELECT COUNT(*) FROM chunks")).scalar_one()
        if indexed != chunk_count:
            _rebuild_fts_index(session)
        session.commit()
        return True
    except Exception:
        session.rollback()
        return False


def _rebuild_fts_index(session: Session) -> None:
    session.execute(text("DELETE FROM chunks_fts"))
    for record in session.scalars(select(ChunkRecord)).all():
        session.execute(
            text("INSERT INTO chunks_fts(chunk_id, chunk_text) VALUES (:chunk_id, :chunk_text)"),
            {"chunk_id": record.chunk_id, "chunk_text": record.chunk_text},
        )


def _search_sqlite_fts(
    session: Session,
    query_terms: list[str],
    top_k: int,
    document_ids: list[str] | None,
    category: str | None,
    date_from_timestamp: int | None,
    date_to_timestamp: int | None,
) -> list[RetrievedChunk]:
    fts_query = " OR ".join(f'"{term}"' for term in query_terms)
    rows = session.execute(
        text(
            "SELECT chunk_id, bm25(chunks_fts) AS rank "
            "FROM chunks_fts WHERE chunks_fts MATCH :query "
            "ORDER BY rank LIMIT :limit"
        ),
        {"query": fts_query, "limit": max(top_k * 10, 25)},
    ).all()
    if not rows:
        return []
    rank_by_chunk_id = {row.chunk_id: float(row.rank) for row in rows}
    stmt = select(ChunkRecord).where(ChunkRecord.chunk_id.in_(rank_by_chunk_id))
    if document_ids:
        stmt = stmt.where(ChunkRecord.document_id.in_(document_ids))
    if category:
        stmt = stmt.where(ChunkRecord.category == category)
    if date_from_timestamp is not None:
        stmt = stmt.where(ChunkRecord.document_timestamp >= date_from_timestamp)
    if date_to_timestamp is not None:
        stmt = stmt.where(ChunkRecord.document_timestamp <= date_to_timestamp)
    records = session.scalars(stmt).all()
    if not records:
        return []
    best_rank = min(rank_by_chunk_id.values())
    worst_rank = max(rank_by_chunk_id.values())
    span = max(1e-9, worst_rank - best_rank)
    candidates = []
    for record in records:
        # SQLite bm25 returns lower scores for better matches. Convert to 0-1 higher-is-better.
        lexical_score = 1.0 if span <= 1e-9 else 1.0 - ((rank_by_chunk_id[record.chunk_id] - best_rank) / span)
        candidates.append(_to_retrieved_chunk(record, lexical_score))
    return sorted(candidates, key=lambda item: item.lexical_score, reverse=True)[:top_k]


def _filtered_chunk_records(
    session: Session,
    document_ids: list[str] | None,
    category: str | None,
    date_from_timestamp: int | None,
    date_to_timestamp: int | None,
) -> list[ChunkRecord]:
    stmt = select(ChunkRecord)
    if document_ids:
        stmt = stmt.where(ChunkRecord.document_id.in_(document_ids))
    if category:
        stmt = stmt.where(ChunkRecord.category == category)
    if date_from_timestamp is not None:
        stmt = stmt.where(ChunkRecord.document_timestamp >= date_from_timestamp)
    if date_to_timestamp is not None:
        stmt = stmt.where(ChunkRecord.document_timestamp <= date_to_timestamp)
    return list(session.scalars(stmt).all())


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
