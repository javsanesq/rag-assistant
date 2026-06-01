from __future__ import annotations


def score_ranked_ids(expected_ids: list[str], actual_ids: list[str], top_k: int, should_answer: bool) -> dict:
    expected = set(expected_ids)
    actual = actual_ids[:top_k]
    abstained = len(actual) == 0
    if not should_answer:
        value = 1.0 if abstained else 0.0
        return {"precision_at_k": value, "hit_rate": value, "recall_at_k": value, "mrr": value}

    hits = [item for item in actual if item in expected]
    first_rank = next((index + 1 for index, item in enumerate(actual) if item in expected), None)
    return {
        "precision_at_k": len(hits) / max(1, top_k),
        "hit_rate": 1.0 if hits else 0.0,
        "recall_at_k": len(set(hits)) / max(1, len(expected)),
        "mrr": 1 / first_rank if first_rank else 0.0,
    }


def answer_contains_expected(answer: str, expected_fragments: list[str]) -> bool:
    if not expected_fragments:
        return True
    normalized = answer.lower()
    return all(fragment.lower() in normalized for fragment in expected_fragments)


def mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)
