from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ChunkingConfig(BaseModel):
    chunker_type: str = "recursive"
    chunk_size: int = 700
    chunk_overlap: int = 120

    @field_validator("chunker_type")
    @classmethod
    def validate_chunker_type(cls, value: str) -> str:
        allowed = {"recursive", "markdown", "sentence"}
        if value not in allowed:
            raise ValueError(f"chunker_type must be one of {sorted(allowed)}")
        return value


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
    dense_score: float
    lexical_score: float = 0.0
    final_score: float
    chunk_id: str
    chunk_index: int


class QueryRequest(BaseModel):
    question: str
    document_ids: list[str] = Field(default_factory=list)
    category: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    top_k: int | None = None
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    alpha: float = 0.75
    include_trace: bool = False

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question is required.")
        return stripped

    @model_validator(mode="after")
    def validate_query(self):
        if self.top_k is not None and not 1 <= self.top_k <= 50:
            raise ValueError("top_k must be between 1 and 50.")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be between 0 and 1.")
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must be before date_to.")
        return self


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    applied_filters: dict[str, Any]
    metrics: dict[str, Any]
    trace: dict[str, Any] | None = None
    grounded: bool = False
    used_citation_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


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
    leased_until: datetime | None = None
    progress: int = 0
    attempts: int = 0
    max_attempts: int = 3
    error_code: str | None = None
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

    @field_validator("dataset_name")
    @classmethod
    def validate_dataset_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("dataset_name is required.")
        if "/" in stripped or "\\" in stripped or ".." in stripped:
            raise ValueError("dataset_name must be a JSONL filename, not a path.")
        if not stripped.endswith(".jsonl"):
            raise ValueError("dataset_name must end with .jsonl.")
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        if any(char not in allowed for char in stripped):
            raise ValueError("dataset_name may only contain letters, numbers, dots, underscores, and hyphens.")
        return stripped


class HealthResponse(BaseModel):
    status: Literal["ok"]
    checks: dict[str, str]
