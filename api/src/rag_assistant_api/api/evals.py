from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from rag_assistant_api.domain.schemas import EvalRunRequest, JobResponse

router = APIRouter(prefix="/api/v1/evals", tags=["evaluations"])


@router.post("/runs", response_model=JobResponse)
def start_eval(request: Request, body: EvalRunRequest, background_tasks: BackgroundTasks) -> JobResponse:
    job_id = request.app.state.evaluation_service.queue_run(body)
    background_tasks.add_task(request.app.state.evaluation_service.run_evaluation, job_id, body)
    return request.app.state.job_service.get_job(job_id)


@router.get("/runs", response_model=list[JobResponse])
def list_eval_runs(request: Request) -> list[JobResponse]:
    return request.app.state.job_service.list_jobs(job_type="evaluation")


@router.get("/runs/{run_id}", response_model=JobResponse)
def get_eval_run(run_id: str, request: Request) -> JobResponse:
    job = request.app.state.job_service.get_job(run_id)
    if not job or job.job_type != "evaluation":
        raise HTTPException(status_code=404, detail="Evaluation run not found.")
    return job
