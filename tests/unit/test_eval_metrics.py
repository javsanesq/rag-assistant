from datetime import datetime, timedelta, timezone

from rag_assistant_api.adapters.llm import MockLLMProvider
from rag_assistant_api.core.config import Settings
from rag_assistant_api.domain.models import JobRecord
from rag_assistant_api.domain.schemas import QueryResponse
from rag_assistant_api.services.jobs import JobService
from rag_assistant_api.services.evaluation import EvaluationService


def test_mock_faithfulness_rubric_returns_expected_shape():
    provider = MockLLMProvider()
    result = provider.judge_faithfulness(
        "What is the refund window?",
        "Grounded answer: Refunds for annual plans are available within 30 calendar days.",
        [{"excerpt": "Refunds for annual plans are available within 30 calendar days of purchase."}],
    )
    assert set(result) == {"score", "rationale"}
    assert 1 <= result["score"] <= 5


def test_eval_dataset_allows_no_answer_rows(tmp_path):
    dataset_dir = tmp_path / "evals"
    dataset_dir.mkdir()
    (dataset_dir / "no_answer.jsonl").write_text(
        '{"id":"missing","question":"What is missing?","expected_document_ids":[],"should_answer":false}\n',
        encoding="utf-8",
    )
    settings = Settings(eval_dataset_dir=dataset_dir)
    service = EvaluationService(settings, query_service=None, job_service=None, llm_provider=MockLLMProvider())

    rows = service._load_dataset("no_answer.jsonl")

    assert rows[0]["should_answer"] is False
    assert rows[0]["expected_document_ids"] == []


def test_eval_dataset_requires_expected_docs_for_answer_rows(tmp_path):
    dataset_dir = tmp_path / "evals"
    dataset_dir.mkdir()
    (dataset_dir / "invalid.jsonl").write_text(
        '{"id":"missing","question":"What is missing?","expected_document_ids":[]}\n',
        encoding="utf-8",
    )
    settings = Settings(eval_dataset_dir=dataset_dir)
    service = EvaluationService(settings, query_service=None, job_service=None, llm_provider=MockLLMProvider())

    try:
        service._load_dataset("invalid.jsonl")
    except ValueError as exc:
        assert "when should_answer is true" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_evaluation_heartbeats_job_before_each_example(client, tmp_path):
    dataset_dir = tmp_path / "evals"
    dataset_dir.mkdir()
    (dataset_dir / "heartbeat.jsonl").write_text(
        '{"id":"missing","question":"What is missing?","expected_document_ids":[],"should_answer":false}\n',
        encoding="utf-8",
    )
    settings = Settings(eval_dataset_dir=dataset_dir)
    job_service = JobService(client.app.state.session_factory, lease_seconds=60)
    job = job_service.create_job("evaluation", {"dataset_name": "heartbeat.jsonl"}, dataset_name="heartbeat.jsonl")
    with client.app.state.session_factory() as session:
        record = session.get(JobRecord, job.id)
        record.leased_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.add(record)
        session.commit()

    service = EvaluationService(
        settings,
        query_service=HeartbeatAssertingQueryService(job_service, job.id),
        job_service=job_service,
        llm_provider=MockLLMProvider(),
    )

    service.run_evaluation(job.id, request=type("Request", (), {"dataset_name": "heartbeat.jsonl", "top_k": None})())

    completed = job_service.get_job(job.id)
    assert completed.status == "completed"


class HeartbeatAssertingQueryService:
    def __init__(self, job_service: JobService, job_id: str) -> None:
        self.job_service = job_service
        self.job_id = job_id

    def answer_question(self, _request) -> QueryResponse:
        job = self.job_service.get_job(self.job_id)
        leased_until = job.leased_until if job.leased_until.tzinfo else job.leased_until.replace(tzinfo=timezone.utc)
        assert leased_until > datetime.now(timezone.utc)
        return QueryResponse(
            answer="I do not have enough evidence in the indexed documents to answer that.",
            citations=[],
            applied_filters={},
            metrics={},
            grounded=False,
            used_citation_ids=[],
            warnings=[],
        )
