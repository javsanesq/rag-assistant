from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import case, or_, select, update
from sqlalchemy.orm import Session, sessionmaker

from rag_assistant_api.domain.models import JobRecord
from rag_assistant_api.domain.schemas import JobResponse

logger = logging.getLogger(__name__)


class JobService:
    def __init__(self, session_factory: sessionmaker[Session], lease_seconds: int = 300) -> None:
        self.session_factory = session_factory
        self.lease_seconds = lease_seconds

    def create_job(
        self,
        job_type: str,
        payload: dict,
        document_id: str | None = None,
        dataset_name: str | None = None,
        result: dict | None = None,
        max_attempts: int = 3,
    ) -> JobRecord:
        with self.session_factory() as session:
            job = JobRecord(
                id=str(uuid4()),
                job_type=job_type,
                status="queued",
                document_id=document_id,
                dataset_name=dataset_name,
                payload_json=json.dumps(payload, default=str),
                result_json=json.dumps(result or {}, default=str),
                max_attempts=max_attempts,
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            return job

    def mark_running(self, job_id: str) -> None:
        now = datetime.now(timezone.utc)
        self._update(job_id, status="running", started_at=now)

    def mark_completed(self, job_id: str, result: dict) -> None:
        self._update(
            job_id,
            status="completed",
            completed_at=datetime.now(timezone.utc),
            progress=100,
            leased_until=None,
            result_json=json.dumps(result, default=str),
        )

    def mark_failed(self, job_id: str, error_message: str, error_code: str = "JOB_FAILED") -> None:
        with self.session_factory() as session:
            job = session.get(JobRecord, job_id)
            if not job:
                return
            job.error_code = error_code
            job.error_message = error_message
            job.leased_until = None
            if job.attempts < job.max_attempts:
                job.status = "queued"
                job.completed_at = None
            else:
                job.status = "failed"
                job.completed_at = datetime.now(timezone.utc)
            session.add(job)
            session.commit()

    def update_progress(self, job_id: str, progress: int, result: dict | None = None) -> None:
        changes = {
            "progress": max(0, min(100, progress)),
            "leased_until": datetime.now(timezone.utc) + timedelta(seconds=self.lease_seconds),
        }
        if result is not None:
            changes["result_json"] = json.dumps(result, default=str)
        self._update(job_id, **changes)

    def claim_next(self, job_types: list[str] | None = None) -> JobRecord | None:
        now = datetime.now(timezone.utc)
        leased_until = now + timedelta(seconds=self.lease_seconds)
        with self.session_factory() as session:
            stmt = (
                select(JobRecord.id)
                .where(JobRecord.status.in_(["queued", "running"]))
                .where(JobRecord.attempts < JobRecord.max_attempts)
                .where(or_(JobRecord.leased_until.is_(None), JobRecord.leased_until < now))
                .order_by(JobRecord.created_at.asc())
                .limit(10)
            )
            if job_types:
                stmt = stmt.where(JobRecord.job_type.in_(job_types))
            candidate_ids = list(session.scalars(stmt).all())

            for candidate_id in candidate_ids:
                claim_stmt = (
                    update(JobRecord)
                    .where(JobRecord.id == candidate_id)
                    .where(JobRecord.status.in_(["queued", "running"]))
                    .where(JobRecord.attempts < JobRecord.max_attempts)
                    .where(or_(JobRecord.leased_until.is_(None), JobRecord.leased_until < now))
                    .values(
                        status="running",
                        started_at=case(
                            (JobRecord.started_at.is_(None), now),
                            else_=JobRecord.started_at,
                        ),
                        attempts=case(
                            (JobRecord.status == "queued", JobRecord.attempts + 1),
                            else_=JobRecord.attempts,
                        ),
                        leased_until=leased_until,
                        error_code=None,
                        error_message=None,
                    )
                    .execution_options(synchronize_session=False)
                )
                result = session.execute(claim_stmt)
                if result.rowcount != 1:
                    session.rollback()
                    continue

                session.commit()
                job = session.get(JobRecord, candidate_id)
                if not job:
                    return None
                logger.info("Job claimed", extra={"job_id": job.id, "event": "job_claimed"})
                return job
            return None

    def retry_job(self, job_id: str) -> JobResponse | None:
        with self.session_factory() as session:
            job = session.get(JobRecord, job_id)
            if not job:
                return None
            if job.status != "failed":
                return self._serialize(job)
            job.status = "queued"
            job.progress = 0
            job.attempts = 0
            job.error_code = None
            job.error_message = None
            job.completed_at = None
            job.leased_until = None
            session.add(job)
            session.commit()
            session.refresh(job)
            return self._serialize(job)

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
            leased_until=job.leased_until,
            progress=job.progress,
            attempts=job.attempts,
            max_attempts=job.max_attempts,
            error_code=job.error_code,
            error_message=job.error_message,
            payload=json.loads(job.payload_json or "{}"),
            result=json.loads(job.result_json or "{}"),
        )
