from rag_assistant_api.adapters.llm import MockLLMProvider


def test_mock_faithfulness_rubric_returns_expected_shape():
    provider = MockLLMProvider()
    result = provider.judge_faithfulness(
        "What is the refund window?",
        "Grounded answer: Refunds for annual plans are available within 30 calendar days.",
        [{"excerpt": "Refunds for annual plans are available within 30 calendar days of purchase."}],
    )
    assert set(result) == {"score", "rationale"}
    assert 1 <= result["score"] <= 5
