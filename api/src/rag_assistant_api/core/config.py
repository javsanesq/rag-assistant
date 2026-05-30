from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    database_url: str | None = None

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_assistant"
    qdrant_vector_size: int = 384
    qdrant_location: str | None = None

    top_k: int = 5
    default_chunker_type: str = "recursive"
    default_chunk_size: int = 700
    default_chunk_overlap: int = 120
    max_chunk_size: int = 1400

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
