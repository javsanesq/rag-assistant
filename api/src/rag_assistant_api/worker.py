from __future__ import annotations

import logging
import time

from rag_assistant_api.core.runtime import build_runtime, close_runtime
from rag_assistant_api.domain.schemas import ChunkingConfig, EvalRunRequest
from rag_assistant_api.services.documents import FilePayload

logger = logging.getLogger(__name__)


def run_once(runtime=None) -> bool:
    owns_runtime = runtime is None
    runtime = runtime or build_runtime()
    try:
        job = runtime.job_service.claim_next(["ingestion", "evaluation"])
        if not job:
            return False
        payload = _payload(job)
        try:
            if job.job_type == "ingestion":
                _run_ingestion(runtime, job.id, payload)
            elif job.job_type == "evaluation":
                runtime.evaluation_service.run_evaluation(job.id, EvalRunRequest(**payload))
            else:
                runtime.job_service.mark_failed(job.id, f"Unsupported job type: {job.job_type}", "UNSUPPORTED_JOB_TYPE")
            return True
        except Exception as exc:  # pragma: no cover - defensive guard for worker process
            logger.exception("Worker job failed", extra={"job_id": job.id})
            runtime.job_service.mark_failed(job.id, str(exc), "WORKER_JOB_FAILED")
            return True
    finally:
        if owns_runtime:
            close_runtime(runtime)


def run_forever() -> None:
    runtime = build_runtime()
    try:
        while True:
            processed = run_once(runtime)
            if not processed:
                time.sleep(runtime.settings.worker_poll_seconds)
    finally:
        close_runtime(runtime)


def _run_ingestion(runtime, job_id: str, payload: dict) -> None:
    chunking = ChunkingConfig(**payload["chunking"])
    metadata = payload.get("metadata", {})
    if payload.get("source") == "files":
        files = [FilePayload(**item) for item in payload.get("files", [])]
        runtime.document_service.process_file_batch(job_id, files, metadata, chunking)
        return
    if payload.get("source") == "urls":
        runtime.document_service.process_url_batch(job_id, payload.get("sources", []), metadata, chunking)
        return
    runtime.job_service.mark_failed(job_id, "Unsupported ingestion source.", "UNSUPPORTED_INGESTION_SOURCE")


def _payload(job) -> dict:
    import json

    return json.loads(job.payload_json or "{}")


if __name__ == "__main__":
    run_forever()
