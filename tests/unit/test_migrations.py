from __future__ import annotations

from pathlib import Path

from sqlalchemy import inspect, text

from rag_assistant_api.core.db import Base, build_engine, init_db
from rag_assistant_api.core.migrations import _sqlite_lock_path
from rag_assistant_api.domain import models  # noqa: F401


def test_init_db_runs_alembic_migrations_for_fresh_sqlite(tmp_path: Path):
    database_url = f"sqlite:///{tmp_path / 'migrated.db'}"
    engine = build_engine(database_url)

    init_db(engine, database_url)

    inspector = inspect(engine)
    assert {"documents", "jobs", "chunks", "alembic_version"}.issubset(set(inspector.get_table_names()))
    job_columns = {column["name"] for column in inspector.get_columns("jobs")}
    assert {"progress", "attempts", "max_attempts", "leased_until"}.issubset(job_columns)

    with engine.connect() as connection:
        version = connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()

    assert version == "20260530_0001"
    engine.dispose()


def test_sqlite_migration_lock_path_uses_database_file(tmp_path: Path):
    db_path = tmp_path / "app.db"

    lock_path = _sqlite_lock_path(f"sqlite:///{db_path}")

    assert lock_path == tmp_path / "app.db.migrate.lock"
    assert _sqlite_lock_path("sqlite:///:memory:") is None
    assert _sqlite_lock_path("postgresql+psycopg://user:pass@localhost/db") is None


def test_init_db_stamps_existing_unversioned_current_schema(tmp_path: Path):
    database_url = f"sqlite:///{tmp_path / 'existing.db'}"
    engine = build_engine(database_url)
    Base.metadata.create_all(bind=engine)

    init_db(engine, database_url)

    with engine.connect() as connection:
        version = connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()

    assert version == "20260530_0001"
    engine.dispose()
