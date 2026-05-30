from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from qdrant_client.http.models import PointStruct
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from rag_assistant_api.adapters.embeddings import EmbeddingProvider
from rag_assistant_api.adapters.parsers import ParsedContent, parse_file_bytes
from rag_assistant_api.adapters.url_loader import expand_urls, fetch_url_content
from rag_assistant_api.adapters.vector_store import QdrantVectorStore
from rag_assistant_api.core.config import Settings
from rag_assistant_api.domain.models import DocumentRecord
from rag_assistant_api.domain.schemas import ChunkingConfig, DocumentResponse, DocumentsListResponse
from rag_assistant_api.services.chunking import chunk_text, validate_chunking
from rag_assistant_api.services.jobs import JobService
from rag_assistant_api.services.metadata import merge_metadata, normalize_document_date, to_timestamp

logger = logging.getLogger(__name__)


@dataclass
class FilePayload:
    filename: str
    content: bytes


class DocumentService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        vector_store: QdrantVectorStore,
        embedding_provider: EmbeddingProvider,
        job_service: JobService,
        settings: Settings,
    ) -> None:
        self.session_factory = session_factory
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.job_service = job_service
        self.settings = settings

    def queue_file_ingest(self, files: list[FilePayload], metadata: dict[str, Any], chunking: ChunkingConfig) -> str:
        job = self.job_service.create_job(
            "ingestion",
            {
                "source": "files",
                "filenames": [item.filename for item in files],
                "metadata": metadata,
                "chunking": chunking.model_dump(),
            },
        )
        return job.id

    def queue_url_ingest(self, request_payload: dict[str, Any]) -> str:
        job = self.job_service.create_job("ingestion", request_payload)
        return job.id

    def process_file_batch(self, job_id: str, files: list[FilePayload], metadata: dict[str, Any], chunking: ChunkingConfig) -> None:
        self.job_service.mark_running(job_id)
        try:
            created_documents = []
            for file in files:
                parsed = parse_file_bytes(file.filename, file.content)
                created_documents.append(self._ingest_parsed_content(parsed, metadata, chunking, job_id))
            self.job_service.mark_completed(job_id, {"documents": created_documents, "total": len(created_documents)})
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("File ingestion failed", extra={"job_id": job_id})
            self.job_service.mark_failed(job_id, str(exc))

    def process_url_batch(self, job_id: str, sources: list[str], metadata: dict[str, Any], chunking: ChunkingConfig) -> None:
        self.job_service.mark_running(job_id)
        try:
            created_documents = []
            for source in sources:
                parsed = fetch_url_content(source)
                created_documents.append(self._ingest_parsed_content(parsed, metadata, chunking, job_id))
            self.job_service.mark_completed(job_id, {"documents": created_documents, "total": len(created_documents)})
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("URL ingestion failed", extra={"job_id": job_id})
            self.job_service.mark_failed(job_id, str(exc))

    def list_documents(self) -> DocumentsListResponse:
        with self.session_factory() as session:
            documents = session.scalars(select(DocumentRecord).order_by(DocumentRecord.created_at.desc())).all()
            categories = [item for item, in session.execute(select(DocumentRecord.category).where(DocumentRecord.category.is_not(None)).distinct())]
        serialized = [self._serialize_document(item) for item in documents]
        dates = [item.document_date for item in serialized if item.document_date]
        facets = {
            "categories": sorted([item for item in categories if item]),
            "document_count": len(serialized),
            "date_bounds": {
                "min": min(dates).isoformat() if dates else None,
                "max": max(dates).isoformat() if dates else None,
            },
        }
        return DocumentsListResponse(documents=serialized, facets=facets)

    def delete_document(self, document_id: str) -> bool:
        with self.session_factory() as session:
            record = session.get(DocumentRecord, document_id)
            if not record:
                return False
            session.delete(record)
            session.commit()
        self.vector_store.delete_document(document_id)
        return True

    def expand_urls(self, url: str | None, urls: list[str], sitemap_url: str | None) -> list[str]:
        return expand_urls(url, urls, sitemap_url)

    def _ingest_parsed_content(
        self,
        parsed: ParsedContent,
        manual_metadata: dict[str, Any],
        chunking: ChunkingConfig,
        job_id: str,
    ) -> dict[str, Any]:
        merged_metadata = merge_metadata(manual_metadata, parsed.metadata)
        document_date = normalize_document_date(merged_metadata)
        category = merged_metadata.get("category")
        document_id = self._next_document_id(parsed.title, parsed.source_uri)
        with self.session_factory() as session:
            record = DocumentRecord(
                document_id=document_id,
                title=parsed.title,
                source_type=parsed.source_type,
                source_uri=parsed.source_uri,
                category=category,
                document_date=document_date,
                document_timestamp=to_timestamp(document_date),
                status="processing",
                metadata_json=json.dumps(merged_metadata, default=str),
                job_id=job_id,
            )
            session.add(record)
            session.commit()

        effective_chunking = validate_chunking(chunking, self.settings.max_chunk_size)
        chunks = chunk_text(parsed.text, effective_chunking)
        embeddings = self.embedding_provider.embed_texts([chunk.text for chunk in chunks]) if chunks else []
        points = []
        for chunk, vector in zip(chunks, embeddings):
            points.append(
                PointStruct(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload={
                        "document_id": document_id,
                        "title": parsed.title,
                        "source_uri": parsed.source_uri,
                        "source_type": parsed.source_type,
                        "category": category,
                        "document_date": document_date.isoformat() if document_date else None,
                        "document_timestamp": to_timestamp(document_date),
                        "chunk_index": chunk.chunk_index,
                        "chunk_text": chunk.text,
                        "metadata": merged_metadata,
                    },
                )
            )
        self.vector_store.upsert_chunks(points)

        with self.session_factory() as session:
            record = session.get(DocumentRecord, document_id)
            if record:
                record.status = "ready"
                record.chunk_count = len(chunks)
                record.summary = chunks[0].text[:300] if chunks else None
                session.add(record)
                session.commit()

        logger.info(
            "Document ingested",
            extra={"event": "document_ingested", "document_id": document_id, "job_id": job_id},
        )
        return {"document_id": document_id, "title": parsed.title, "chunk_count": len(chunks)}

    def _next_document_id(self, title: str, source_uri: str) -> str:
        candidate = _slugify(Path(source_uri).stem if source_uri and "://" not in source_uri else title)
        with self.session_factory() as session:
            existing = session.get(DocumentRecord, candidate)
            if not existing:
                return candidate
            counter = 2
            while session.get(DocumentRecord, f"{candidate}-{counter}"):
                counter += 1
        return f"{candidate}-{counter}"

    def _serialize_document(self, record: DocumentRecord) -> DocumentResponse:
        return DocumentResponse(
            document_id=record.document_id,
            title=record.title,
            source_type=record.source_type,
            source_uri=record.source_uri,
            category=record.category,
            document_date=record.document_date,
            status=record.status,
            chunk_count=record.chunk_count,
            metadata=json.loads(record.metadata_json or "{}"),
            created_at=record.created_at,
        )


def _slugify(value: str) -> str:
    sanitized = "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")
    while "--" in sanitized:
        sanitized = sanitized.replace("--", "-")
    return sanitized or "document"
