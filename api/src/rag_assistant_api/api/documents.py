from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from rag_assistant_api.domain.schemas import ChunkingConfig, DocumentsListResponse, JobResponse, URLIngestRequest
from rag_assistant_api.services.documents import FilePayload

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("/files", response_model=JobResponse)
async def ingest_files(
    request: Request,
    files: list[UploadFile] = File(...),
    metadata_json: str | None = Form(default=None),
    category: str | None = Form(default=None),
    document_date: str | None = Form(default=None),
    chunker_type: str | None = Form(default=None),
    chunk_size: int | None = Form(default=None),
    chunk_overlap: int | None = Form(default=None),
) -> JobResponse:
    try:
        metadata = json.loads(metadata_json) if metadata_json else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"metadata_json is invalid JSON: {exc.msg}") from exc
    if category:
        metadata["category"] = category
    if document_date:
        metadata["document_date"] = document_date
    payloads, rejected = await _persist_uploads(files, request)
    if not payloads:
        raise HTTPException(status_code=400, detail={"message": "No files accepted.", "rejected_files": rejected})
    chunking = ChunkingConfig(
        chunker_type=chunker_type or request.app.state.settings.default_chunker_type,
        chunk_size=chunk_size or request.app.state.settings.default_chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else request.app.state.settings.default_chunk_overlap,
    )
    job_id = request.app.state.document_service.queue_file_ingest(payloads, metadata, chunking, rejected)
    return request.app.state.job_service.get_job(job_id)


@router.post("/urls", response_model=JobResponse)
def ingest_urls(request: Request, body: URLIngestRequest) -> JobResponse:
    metadata = dict(body.metadata)
    if body.category:
        metadata["category"] = body.category
    if body.document_date:
        metadata["document_date"] = body.document_date.isoformat()
    try:
        sources = request.app.state.document_service.expand_urls(body.url, body.urls, body.sitemap_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    chunking = ChunkingConfig(
        chunker_type=body.chunker_type or request.app.state.settings.default_chunker_type,
        chunk_size=body.chunk_size or request.app.state.settings.default_chunk_size,
        chunk_overlap=body.chunk_overlap if body.chunk_overlap is not None else request.app.state.settings.default_chunk_overlap,
    )
    job_id = request.app.state.document_service.queue_url_ingest({"sources": sources, "metadata": metadata, "chunking": chunking.model_dump()})
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


async def _persist_uploads(files: list[UploadFile], request: Request) -> tuple[list[FilePayload], list[dict]]:
    settings = request.app.state.settings
    batch_dir = settings.uploads_dir / str(uuid4())
    batch_dir.mkdir(parents=True, exist_ok=True)
    accepted: list[FilePayload] = []
    rejected: list[dict] = []
    allowed_extensions = {item.lower() for item in settings.accepted_file_extensions}

    for upload in files:
        filename = Path(upload.filename or "upload").name
        extension = Path(filename).suffix.lower()
        if extension not in allowed_extensions:
            rejected.append({"filename": filename, "reason": f"Unsupported file extension: {extension or 'none'}"})
            continue
        target = batch_dir / f"{uuid4()}-{filename}"
        size = 0
        with target.open("wb") as handle:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_file_bytes:
                    handle.close()
                    target.unlink(missing_ok=True)
                    rejected.append({"filename": filename, "reason": "File exceeds MAX_UPLOAD_FILE_BYTES."})
                    break
                handle.write(chunk)
        if size == 0 and target.exists():
            target.unlink(missing_ok=True)
            rejected.append({"filename": filename, "reason": "File is empty."})
            continue
        if target.exists():
            accepted.append(
                FilePayload(
                    filename=filename,
                    path=str(target),
                    content_type=upload.content_type,
                    size_bytes=size,
                )
            )
    return accepted, rejected
