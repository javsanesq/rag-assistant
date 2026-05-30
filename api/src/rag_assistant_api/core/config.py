from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    database_url: str | None = None
    worker_poll_seconds: float = 2.0
    job_lease_seconds: int = 300

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_assistant"
    qdrant_vector_size: int = 384
    qdrant_location: str | None = None

    top_k: int = 5
    default_chunker_type: str = "recursive"
    default_chunk_size: int = 700
    default_chunk_overlap: int = 120
    max_chunk_size: int = 1400
    max_upload_file_bytes: int = 25 * 1024 * 1024
    max_url_bytes: int = 5 * 1024 * 1024
    max_sitemap_urls: int = 100
    accepted_file_extensions: list[str] = Field(default_factory=lambda: [".pdf", ".docx", ".md", ".markdown"])
    allow_private_urls: bool = False
    url_allowed_domains: list[str] = Field(default_factory=list)
    url_blocked_domains: list[str] = Field(default_factory=list)

    embed_provider: str = "sentence-transformers"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    llm_provider: str = "mock"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    eval_dataset_dir: Path | None = None
    eval_faithfulness_use_llm: bool = False

    @field_validator("api_cors_origins", mode="before")
    @classmethod
    def split_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("accepted_file_extensions", "url_allowed_domains", "url_blocked_domains", mode="before")
    @classmethod
    def split_csv(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def validate_runtime_settings(self):
        if self.qdrant_vector_size <= 0:
            raise ValueError("QDRANT_VECTOR_SIZE must be positive.")
        if self.top_k <= 0:
            raise ValueError("TOP_K must be positive.")
        if self.default_chunk_overlap >= self.default_chunk_size:
            raise ValueError("DEFAULT_CHUNK_OVERLAP must be smaller than DEFAULT_CHUNK_SIZE.")
        if self.embed_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBED_PROVIDER=openai.")
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
        return self

    @property
    def root_dir(self) -> Path:
        return Path(__file__).resolve().parents[4]

    @property
    def effective_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{self.root_dir / 'data' / 'app.db'}"

    @property
    def effective_eval_dataset_dir(self) -> Path:
        if self.eval_dataset_dir:
            return self.eval_dataset_dir
        return self.root_dir / "evals" / "datasets"

    @property
    def data_dir(self) -> Path:
        return self.root_dir / "data"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"
