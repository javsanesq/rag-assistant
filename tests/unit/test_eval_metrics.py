from rag_assistant_api.adapters.llm import MockLLMProvider
from rag_assistant_api.core.config import Settings
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
