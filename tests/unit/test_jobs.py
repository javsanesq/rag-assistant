from datetime import datetime, timedelta, timezone

from rag_assistant_api.domain.models import JobRecord
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


def test_retry_job_resets_attempts_so_maxed_failed_job_can_be_claimed(client):
    service = JobService(client.app.state.session_factory, lease_seconds=60)
    job = service.create_job("ingestion", {"source": "noop"}, max_attempts=1)

    first_claim = service.claim_next(["ingestion"])
    assert first_claim.id == job.id
    assert first_claim.attempts == 1

    service.mark_failed(job.id, "boom", "TEST_FAILURE")
    assert service.claim_next(["ingestion"]) is None

    retried = service.retry_job(job.id)
    assert retried.status == "queued"
    assert retried.attempts == 0

    second_claim = service.claim_next(["ingestion"])
    assert second_claim.id == job.id
    assert second_claim.status == "running"
    assert second_claim.attempts == 1


def test_claim_next_respects_active_leases_and_reclaims_expired_leases(client):
    service = JobService(client.app.state.session_factory, lease_seconds=60)
    active_job = service.create_job("ingestion", {"source": "active"})
    queued_job = service.create_job("ingestion", {"source": "queued"})

    active_claim = service.claim_next(["ingestion"])
    assert active_claim.id == active_job.id
    assert service.claim_next(["ingestion"]).id == queued_job.id

    assert service.claim_next(["ingestion"]) is None

    with client.app.state.session_factory() as session:
        active_record = session.get(JobRecord, active_job.id)
        assert active_record is not None
        active_record.leased_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.add(active_record)
        session.commit()

    reclaimed = service.claim_next(["ingestion"])
    assert reclaimed.id == active_job.id
    assert reclaimed.attempts == 2
