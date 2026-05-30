from __future__ import annotations

import json
from datetime import date

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile

from rag_assistant_api.domain.schemas import ChunkingConfig, DocumentsListResponse, JobResponse, URLIngestRequest
from rag_assistant_api.services.documents import FilePayload

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("/files", response_model=JobResponse)
async def ingest_files(
    request: Request,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    metadata_json: str | None = Form(default=None),
    category: str | None = Form(default=None),
    document_date: str | None = Form(default=None),
    chunker_type: str | None = Form(default=None),
    chunk_size: int | None = Form(default=None),
    chunk_overlap: int | None = Form(default=None),
) -> JobResponse:
    metadata = json.loads(metadata_json) if metadata_json else {}
    if category:
        metadata["category"] = category
    if document_date:
        metadata["document_date"] = document_date
    payloads = [FilePayload(filename=item.filename or "upload", content=await item.read()) for item in files]
    chunking = ChunkingConfig(
        chunker_type=chunker_type or request.app.state.settings.default_chunker_type,
        chunk_size=chunk_size or request.app.state.settings.default_chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else request.app.state.settings.default_chunk_overlap,
    )
    job_id = request.app.state.document_service.queue_file_ingest(payloads, metadata, chunking)
    background_tasks.add_task(request.app.state.document_service.process_file_batch, job_id, payloads, metadata, chunking)
    return request.app.state.job_service.get_job(job_id)


@router.post("/urls", response_model=JobResponse)
def ingest_urls(request: Request, body: URLIngestRequest, background_tasks: BackgroundTasks) -> JobResponse:
    metadata = dict(body.metadata)
    if body.category:
        metadata["category"] = body.category
    if body.document_date:
        metadata["document_date"] = body.document_date.isoformat()
    sources = request.app.state.document_service.expand_urls(body.url, body.urls, body.sitemap_url)
    chunking = ChunkingConfig(
        chunker_type=body.chunker_type or request.app.state.settings.default_chunker_type,
        chunk_size=body.chunk_size or request.app.state.settings.default_chunk_size,
        chunk_overlap=body.chunk_overlap if body.chunk_overlap is not None else request.app.state.settings.default_chunk_overlap,
    )
    job_id = request.app.state.document_service.queue_url_ingest({"sources": sources, "metadata": metadata, "chunking": chunking.model_dump()})
    background_tasks.add_task(request.app.state.document_service.process_url_batch, job_id, sources, metadata, chunking)
    return request.app.state.job_service.get_job(job_id)


@router.get("", response_model=DocumentsListResponse)
def list_documents(request: Request) -> DocumentsListResponse:
    return request.app.state.document_service.list_documents()


@router.delete("/{document_id}")
def delete_document(document_id: str, request: Request) -> dict:
    deleted = request.app.state.document_service.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"deleted": True, "document_id": document_id}
