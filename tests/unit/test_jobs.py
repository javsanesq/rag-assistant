from rag_assistant_api.services.jobs import JobService


def test_job_claim_and_retry(client):
    service = JobService(client.app.state.session_factory, lease_seconds=60)
    job = service.create_job("ingestion", {"source": "noop"})

    claimed = service.claim_next(["ingestion"])
    assert claimed.id == job.id
    assert claimed.status == "running"
    assert claimed.attempts == 1

    service.mark_failed(job.id, "boom", "TEST_FAILURE")
    failed = service.get_job(job.id)
    assert failed.status == "failed"
    assert failed.error_code == "TEST_FAILURE"

    retried = service.retry_job(job.id)
    assert retried.status == "queued"
    assert retried.error_code is None
