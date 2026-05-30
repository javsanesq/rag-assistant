from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ChunkingConfig(BaseModel):
    chunker_type: str = "recursive"
    chunk_size: int = 700
    chunk_overlap: int = 120


class URLIngestRequest(BaseModel):
    url: str | None = None
    urls: list[str] = Field(default_factory=list)
    sitemap_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    category: str | None = None
    document_date: date | None = None
    chunker_type: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None

    @model_validator(mode="after")
    def validate_sources(self):
        if not any([self.url, self.urls, self.sitemap_url]):
            raise ValueError("Provide at least one of url, urls, or sitemap_url.")
        return self


class Citation(BaseModel):
    document_id: str
    title: str
    source_uri: str
    category: str | None = None
    document_date: date | None = None
    excerpt: str
    score: float
    chunk_id: str
    chunk_index: int


class QueryRequest(BaseModel):
    question: str
    document_ids: list[str] = Field(default_factory=list)
    category: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    top_k: int | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    applied_filters: dict[str, Any]
    metrics: dict[str, Any]


class JobResponse(BaseModel):
    id: str
    job_type: str
    status: str
    document_id: str | None = None
    dataset_name: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    document_id: str
    title: str
    source_type: str
    source_uri: str
    category: str | None = None
    document_date: date | None = None
    status: str
    chunk_count: int
    metadata: dict[str, Any]
    created_at: datetime


class DocumentsListResponse(BaseModel):
    documents: list[DocumentResponse]
    facets: dict[str, Any]


class EvalRunRequest(BaseModel):
    dataset_name: str
    top_k: int | None = None
    filters: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    checks: dict[str, str]
