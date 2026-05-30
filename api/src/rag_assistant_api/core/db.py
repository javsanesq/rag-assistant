from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


def build_engine(database_url: str):
    connect_args = {}
    if database_url.startswith("sqlite"):
        sqlite_path = database_url.replace("sqlite:///", "", 1)
        if sqlite_path and sqlite_path != ":memory:":
            Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        connect_args["check_same_thread"] = False
    return create_engine(database_url, future=True, connect_args=connect_args)


def build_session_factory(engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_db(engine) -> None:
    from rag_assistant_api.domain.models import DocumentRecord, JobRecord  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _ensure_job_columns(engine)


def _ensure_job_columns(engine) -> None:
    inspector = inspect(engine)
    if "jobs" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("jobs")}
    dialect = engine.dialect.name
    column_specs = {
        "progress": "INTEGER DEFAULT 0",
        "attempts": "INTEGER DEFAULT 0",
        "max_attempts": "INTEGER DEFAULT 3",
        "error_code": "VARCHAR(64)",
        "leased_until": "DATETIME" if dialect == "sqlite" else "TIMESTAMP",
    }
    with engine.begin() as connection:
        for column, spec in column_specs.items():
            if column not in existing:
                connection.execute(text(f"ALTER TABLE jobs ADD COLUMN {column} {spec}"))
