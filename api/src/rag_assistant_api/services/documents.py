from __future__ import annotations

import json
import logging
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client.http.models import PointStruct
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from rag_assistant_api.adapters.embeddings import EmbeddingProvider
from rag_assistant_api.adapters.parsers import ParsedContent, parse_file_bytes
from rag_assistant_api.adapters.url_loader import expand_urls, fetch_url_content
from rag_assistant_api.adapters.vector_store import QdrantVectorStore
from rag_assistant_api.core.config import Settings
from rag_assistant_api.domain.models import ChunkRecord, DocumentRecord
from rag_assistant_api.domain.schemas import ChunkingConfig, DocumentResponse, DocumentsListResponse
from rag_assistant_api.services.chunking import chunk_text, validate_chunking
from rag_assistant_api.services.jobs import JobService
from rag_assistant_api.services.metadata import merge_metadata, normalize_document_date, to_timestamp

logger = logging.getLogger(__name__)


@dataclass
class FilePayload:
    filename: str
    path: str
    content_type: str | None = None
    size_bytes: int = 0


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

    def queue_file_ingest(
        self,
        files: list[FilePayload],
        metadata: dict[str, Any],
        chunking: ChunkingConfig,
        rejected_files: list[dict[str, Any]] | None = None,
    ) -> str:
        job = self.job_service.create_job(
            "ingestion",
            {
                "source": "files",
                "files": [file.__dict__ for file in files],
                "metadata": metadata,
                "chunking": chunking.model_dump(),
            },
            result={
                "accepted_files": [{"filename": item.filename, "size_bytes": item.size_bytes} for item in files],
                "rejected_files": rejected_files or [],
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
            failed_documents = []
            total = max(1, len(files))
            for index, file in enumerate(files, start=1):
                try:
                    content = Path(file.path).read_bytes()
                    parsed = parse_file_bytes(file.filename, content)
                    if not parsed.text.strip():
                        raise ValueError("No extractable text found.")
                    created_documents.append(self._ingest_parsed_content(parsed, metadata, chunking, job_id, content))
                except Exception as exc:
                    failed_documents.append({"filename": file.filename, "error": str(exc)})
                self.job_service.update_progress(job_id, int((index / total) * 90))
            self.job_service.mark_completed(
                job_id,
                {
                    "documents": created_documents,
                    "failed_documents": failed_documents,
                    "total": len(created_documents),
                    "failed_total": len(failed_documents),
                },
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("File ingestion failed", extra={"job_id": job_id})
            self.job_service.mark_failed(job_id, str(exc), error_code="INGESTION_FAILED")

    def process_url_batch(self, job_id: str, sources: list[str], metadata: dict[str, Any], chunking: ChunkingConfig) -> None:
        self.job_service.mark_running(job_id)
        try:
            created_documents = []
            for source in sources:
                parsed = fetch_url_content(source, self.settings)
                if not parsed.text.strip():
                    raise ValueError(f"No extractable text found at {source}")
                created_documents.append(self._ingest_parsed_content(parsed, metadata, chunking, job_id, parsed.text.encode("utf-8")))
                self.job_service.update_progress(job_id, int((len(created_documents) / max(1, len(sources))) * 90))
            self.job_service.mark_completed(job_id, {"documents": created_documents, "total": len(created_documents)})
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("URL ingestion failed", extra={"job_id": job_id})
            self.job_service.mark_failed(job_id, str(exc), error_code="URL_INGESTION_FAILED")

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
            if not session.get(DocumentRecord, document_id):
                return False
        self.vector_store.delete_document(document_id)
        with self.session_factory() as session:
            record = session.get(DocumentRecord, document_id)
            if not record:
                return False
            session.execute(delete(ChunkRecord).where(ChunkRecord.document_id == document_id))
            session.delete(record)
            session.commit()
        return True

    def expand_urls(self, url: str | None, urls: list[str], sitemap_url: str | None) -> list[str]:
        return expand_urls(url, urls, sitemap_url, self.settings)

    def _ingest_parsed_content(
        self,
        parsed: ParsedContent,
        manual_metadata: dict[str, Any],
        chunking: ChunkingConfig,
        job_id: str,
        raw_content: bytes,
    ) -> dict[str, Any]:
        merged_metadata = merge_metadata(manual_metadata, parsed.metadata)
        document_date = normalize_document_date(merged_metadata)
        category = merged_metadata.get("category")
        source_hash = hashlib.sha256(raw_content).hexdigest()
        merged_metadata["source_hash"] = source_hash
        serialized_metadata = json.loads(json.dumps(merged_metadata, default=str))
        document_id = self._existing_document_id_for_hash(source_hash, parsed.source_uri)
        if document_id:
            self.delete_document(document_id)
        else:
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
                metadata_json=json.dumps(serialized_metadata),
                job_id=job_id,
            )
            session.add(record)
            session.commit()

        try:
            effective_chunking = validate_chunking(chunking, self.settings.max_chunk_size)
            chunks = chunk_text(parsed.text, effective_chunking)
            embeddings = self.embedding_provider.embed_texts([chunk.text for chunk in chunks]) if chunks else []
            points = []
            chunk_records = []
            for chunk, vector in zip(chunks, embeddings):
                lexical_terms = _tokenize(chunk.text)
                chunk_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
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
                            "chunk_hash": chunk_hash,
                            "source_hash": source_hash,
                            "lexical_terms": lexical_terms,
                            "metadata": serialized_metadata,
                        },
                    )
                )
                chunk_records.append(
                    ChunkRecord(
                        chunk_id=chunk.chunk_id,
                        document_id=document_id,
                        title=parsed.title,
                        source_uri=parsed.source_uri,
                        source_type=parsed.source_type,
                        category=category,
                        document_date=document_date,
                        document_timestamp=to_timestamp(document_date),
                        chunk_index=chunk.chunk_index,
                        chunk_text=chunk.text,
                        lexical_terms_json=json.dumps(lexical_terms),
                    )
                )
            self.vector_store.upsert_chunks(points)

            with self.session_factory() as session:
                session.execute(delete(ChunkRecord).where(ChunkRecord.document_id == document_id))
                session.add_all(chunk_records)
                record = session.get(DocumentRecord, document_id)
                if record:
                    record.status = "ready"
                    record.chunk_count = len(chunks)
                    record.summary = chunks[0].text[:300] if chunks else None
                    session.add(record)
                    session.commit()
        except Exception:
            self.vector_store.delete_document(document_id)
            with self.session_factory() as session:
                record = session.get(DocumentRecord, document_id)
                if record:
                    record.status = "failed"
                    record.summary = "Ingestion failed before document became queryable."
                    session.add(record)
                    session.commit()
            raise

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

    def _existing_document_id_for_hash(self, source_hash: str, source_uri: str) -> str | None:
        with self.session_factory() as session:
            records = session.scalars(select(DocumentRecord).where(DocumentRecord.source_uri == source_uri)).all()
        for record in records:
            metadata = json.loads(record.metadata_json or "{}")
            if metadata.get("source_hash") == source_hash:
                return record.document_id
        return None

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


def _tokenize(text: str) -> list[str]:
    return sorted(set(re.findall(r"[a-zA-Z0-9]{3,}", text.lower())))
