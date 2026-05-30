from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, FieldCondition, Filter, MatchAny, MatchValue, PointStruct, Range, VectorParams

from rag_assistant_api.core.config import Settings


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    title: str
    source_uri: str
    category: str | None
    document_date: date | None
    excerpt: str
    score: float
    dense_score: float
    lexical_score: float
    final_score: float
    chunk_index: int


class QdrantVectorStore:
    def __init__(self, settings: Settings, dimensions: int) -> None:
        self.settings = settings
        self.client = QdrantClient(location=settings.qdrant_location) if settings.qdrant_location else QdrantClient(url=settings.qdrant_url)
        self.collection_name = settings.qdrant_collection
        self.dimensions = dimensions
        self.ensure_collection()

    def ensure_collection(self) -> None:
        collections = {item.name for item in self.client.get_collections().collections}
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dimensions, distance=Distance.COSINE),
            )

    def upsert_chunks(self, points: list[PointStruct]) -> None:
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    def delete_document(self, document_id: str) -> None:
        filter_ = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
        self.client.delete(collection_name=self.collection_name, points_selector=filter_)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
        category: str | None = None,
        date_from_timestamp: int | None = None,
        date_to_timestamp: int | None = None,
        query_text: str = "",
        retrieval_mode: str = "hybrid",
        alpha: float = 0.75,
    ) -> list[RetrievedChunk]:
        filter_ = self._build_filter(document_ids, category, date_from_timestamp, date_to_timestamp)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k * 4 if retrieval_mode == "hybrid" else top_k,
            query_filter=filter_,
            with_payload=True,
        )
        hits = response.points
        results: list[RetrievedChunk] = []
        query_terms = _tokenize(query_text)
        for hit in hits:
            payload = hit.payload or {}
            document_date = date.fromisoformat(payload["document_date"]) if payload.get("document_date") else None
            dense_score = float(hit.score)
            dense_for_final = max(0.0, min(1.0, dense_score))
            lexical_score = _lexical_score(query_terms, payload.get("lexical_terms") or [])
            final_score = dense_for_final if retrieval_mode == "dense" else alpha * dense_for_final + (1 - alpha) * lexical_score
            results.append(
                RetrievedChunk(
                    chunk_id=str(hit.id),
                    document_id=str(payload.get("document_id", "")),
                    title=str(payload.get("title", "")),
                    source_uri=str(payload.get("source_uri", "")),
                    category=payload.get("category"),
                    document_date=document_date,
                    excerpt=str(payload.get("chunk_text", "")),
                    score=final_score,
                    dense_score=dense_score,
                    lexical_score=lexical_score,
                    final_score=final_score,
                    chunk_index=int(payload.get("chunk_index", 0)),
                )
            )
        return sorted(results, key=lambda item: item.final_score, reverse=True)[:top_k]

    def healthcheck(self) -> bool:
        return bool(self.client.get_collections().collections or True)

    def close(self) -> None:
        self.client.close()

    def _build_filter(
        self,
        document_ids: list[str] | None,
        category: str | None,
        date_from_timestamp: int | None,
        date_to_timestamp: int | None,
    ) -> Filter | None:
        must = []
        if document_ids:
            must.append(FieldCondition(key="document_id", match=MatchAny(any=document_ids)))
        if category:
            must.append(FieldCondition(key="category", match=MatchValue(value=category)))
        if date_from_timestamp is not None or date_to_timestamp is not None:
            must.append(
                FieldCondition(
                    key="document_timestamp",
                    range=Range(gte=date_from_timestamp, lte=date_to_timestamp),
                )
            )
        return Filter(must=must) if must else None


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-zA-Z0-9]{3,}", text.lower()) if token]


def _lexical_score(query_terms: list[str], chunk_terms: list[str]) -> float:
    if not query_terms or not chunk_terms:
        return 0.0
    query_set = set(query_terms)
    chunk_set = set(chunk_terms)
    return len(query_set & chunk_set) / len(query_set)
