from __future__ import annotations

from pathlib import Path


def _sample_path(name: str) -> Path:
    return Path(__file__).resolve().parents[2] / "samples" / "knowledge" / name


def test_health_ready(client):
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["checks"]["qdrant"] == "ok"


def test_file_ingest_query_and_eval_flow(client):
    files = [
        ("files", ("company-handbook.md", _sample_path("company-handbook.md").read_bytes(), "text/markdown")),
        ("files", ("release-notes.md", _sample_path("release-notes.md").read_bytes(), "text/markdown")),
    ]
    ingest_response = client.post("/api/v1/documents/files", files=files)
    assert ingest_response.status_code == 200
    assert ingest_response.json()["status"] in {"queued", "completed"}

    docs_response = client.get("/api/v1/documents")
    assert docs_response.status_code == 200
    payload = docs_response.json()
    assert payload["facets"]["document_count"] == 2

    query_response = client.post(
        "/api/v1/query",
        json={"question": "What is the refund window for annual plans?", "category": "policy"},
    )
    assert query_response.status_code == 200
    query_payload = query_response.json()
    assert query_payload["citations"]
    assert "30 calendar days" in query_payload["answer"]

    eval_response = client.post("/api/v1/evals/runs", json={"dataset_name": "portfolio_eval.jsonl"})
    assert eval_response.status_code == 200
    run_id = eval_response.json()["id"]

    run_detail = client.get(f"/api/v1/evals/runs/{run_id}")
    assert run_detail.status_code == 200
    assert run_detail.json()["status"] in {"queued", "completed"}
