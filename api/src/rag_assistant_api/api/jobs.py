from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from rag_assistant_api.domain.schemas import JobResponse

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.get("", response_model=list[JobResponse])
def list_jobs(request: Request) -> list[JobResponse]:
    return request.app.state.job_service.list_jobs()


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str, request: Request) -> JobResponse:
    job = request.app.state.job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job
