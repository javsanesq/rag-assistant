from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from rag_assistant_api.domain.models import JobRecord
from rag_assistant_api.domain.schemas import JobResponse


class JobService:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def create_job(self, job_type: str, payload: dict, document_id: str | None = None, dataset_name: str | None = None) -> JobRecord:
        with self.session_factory() as session:
            job = JobRecord(
                id=str(uuid4()),
                job_type=job_type,
                status="queued",
                document_id=document_id,
                dataset_name=dataset_name,
                payload_json=json.dumps(payload),
                result_json="{}",
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            return job

    def mark_running(self, job_id: str) -> None:
        self._update(job_id, status="running", started_at=datetime.now(timezone.utc))

    def mark_completed(self, job_id: str, result: dict) -> None:
        self._update(job_id, status="completed", completed_at=datetime.now(timezone.utc), result_json=json.dumps(result))

    def mark_failed(self, job_id: str, error_message: str) -> None:
        self._update(job_id, status="failed", completed_at=datetime.now(timezone.utc), error_message=error_message)

    def list_jobs(self, job_type: str | None = None) -> list[JobResponse]:
        with self.session_factory() as session:
            stmt = select(JobRecord).order_by(JobRecord.created_at.desc())
            if job_type:
                stmt = stmt.where(JobRecord.job_type == job_type)
            return [self._serialize(job) for job in session.scalars(stmt).all()]

    def get_job(self, job_id: str) -> JobResponse | None:
        with self.session_factory() as session:
            job = session.get(JobRecord, job_id)
            return self._serialize(job) if job else None

    def _update(self, job_id: str, **changes) -> None:
        with self.session_factory() as session:
            job = session.get(JobRecord, job_id)
            if not job:
                return
            for key, value in changes.items():
                setattr(job, key, value)
            session.add(job)
            session.commit()

    def _serialize(self, job: JobRecord) -> JobResponse:
        return JobResponse(
            id=job.id,
            job_type=job.job_type,
            status=job.status,
            document_id=job.document_id,
            dataset_name=job.dataset_name,
            created_at=job.created_at,
            updated_at=job.updated_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
            payload=json.loads(job.payload_json or "{}"),
            result=json.loads(job.result_json or "{}"),
        )
