from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect


BASELINE_REVISION = "20260530_0001"
EXPECTED_TABLES = {"documents", "jobs", "chunks"}
EXPECTED_COLUMNS = {
    "documents": {
        "document_id",
        "title",
        "source_type",
        "source_uri",
        "category",
        "document_date",
        "document_timestamp",
        "status",
        "metadata_json",
        "chunk_count",
        "summary",
        "job_id",
        "created_at",
        "updated_at",
    },
    "jobs": {
        "id",
        "job_type",
        "status",
        "document_id",
        "dataset_name",
        "payload_json",
        "result_json",
        "progress",
        "attempts",
        "max_attempts",
        "error_code",
        "error_message",
        "created_at",
        "updated_at",
        "started_at",
        "completed_at",
        "leased_until",
    },
    "chunks": {
        "chunk_id",
        "document_id",
        "title",
        "source_uri",
        "source_type",
        "category",
        "document_date",
        "document_timestamp",
        "chunk_index",
        "chunk_text",
        "lexical_terms_json",
        "created_at",
    },
}


def run_migrations(engine: Engine, database_url: str) -> None:
    alembic_config = _build_alembic_config(database_url)
    _stamp_existing_current_schema(engine, alembic_config)
    with engine.begin() as connection:
        alembic_config.attributes["connection"] = connection
        command.upgrade(alembic_config, "head")


def _build_alembic_config(database_url: str) -> Config:
    root_dir = Path(__file__).resolve().parents[4]
    api_dir = root_dir / "api"
    config = Config(str(api_dir / "alembic.ini"))
    config.set_main_option("script_location", str(api_dir / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def _stamp_existing_current_schema(engine: Engine, alembic_config: Config) -> None:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if "alembic_version" in tables or not EXPECTED_TABLES.issubset(tables):
        return

    missing_columns = _missing_expected_columns(inspector)
    if missing_columns:
        details = ", ".join(f"{table}: {sorted(columns)}" for table, columns in missing_columns.items())
        raise RuntimeError(
            "Database has unversioned RAG tables but does not match the current schema. "
            f"Missing columns: {details}. Export the data or migrate it manually before startup."
        )

    with engine.begin() as connection:
        alembic_config.attributes["connection"] = connection
        command.stamp(alembic_config, BASELINE_REVISION)
        alembic_config.attributes.pop("connection", None)


def _missing_expected_columns(inspector) -> dict[str, set[str]]:
    missing = {}
    for table, expected_columns in EXPECTED_COLUMNS.items():
        actual_columns = {column["name"] for column in inspector.get_columns(table)}
        table_missing = expected_columns - actual_columns
        if table_missing:
            missing[table] = table_missing
    return missing
