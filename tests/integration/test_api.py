from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from rag_assistant_api.adapters import url_loader
from rag_assistant_api.adapters.parsers import ParsedContent
from rag_assistant_api.domain.models import ChunkRecord
from rag_assistant_api.services import documents as document_service_module
from rag_assistant_api.worker import run_once


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
    assert ingest_response.json()["status"] == "queued"
    assert run_once(client.app.state.runtime) is True

    docs_response = client.get("/api/v1/documents")
    assert docs_response.status_code == 200
    payload = docs_response.json()
    assert payload["facets"]["document_count"] == 2
    with client.app.state.session_factory() as session:
        assert session.scalar(select(ChunkRecord).where(ChunkRecord.document_id == "company-handbook").limit(1)) is not None

    query_response = client.post(
        "/api/v1/query",
        json={"question": "What is the refund window for annual plans?", "category": "policy"},
    )
    assert query_response.status_code == 200
    query_payload = query_response.json()
    assert query_payload["citations"]
    assert "30 calendar days" in query_payload["answer"]
    assert query_payload["grounded"] is True
    assert query_payload["used_citation_ids"] == [query_payload["citations"][0]["chunk_id"]]
    assert "[1]" in query_payload["answer"]
    assert query_payload["citations"][0]["final_score"] >= query_payload["citations"][0]["lexical_score"] * 0

    eval_response = client.post("/api/v1/evals/runs", json={"dataset_name": "portfolio_eval.jsonl"})
    assert eval_response.status_code == 200
    run_id = eval_response.json()["id"]
    assert run_once(client.app.state.runtime) is True

    run_detail = client.get(f"/api/v1/evals/runs/{run_id}")
    assert run_detail.status_code == 200
    assert run_detail.json()["status"] == "completed"
    assert "mrr" in run_detail.json()["result"]["summary"]

    delete_response = client.delete("/api/v1/documents/company-handbook")
    assert delete_response.status_code == 200
    with client.app.state.session_factory() as session:
        assert session.scalar(select(ChunkRecord).where(ChunkRecord.document_id == "company-handbook").limit(1)) is None


def test_invalid_upload_is_rejected(client):
    response = client.post(
        "/api/v1/documents/files",
        files=[("files", ("notes.txt", b"not supported", "text/plain"))],
    )
    assert response.status_code == 400
    assert response.json()["detail"]["rejected_files"][0]["reason"].startswith("Unsupported")


def test_private_url_is_blocked(client):
    response = client.post("/api/v1/documents/urls", json={"url": "http://127.0.0.1:8080"})
    assert response.status_code == 400
    assert "blocked" in response.json()["detail"].lower()


def test_url_ingest_job_completes_and_creates_document(client, monkeypatch):
    def fake_getaddrinfo(hostname, *_args, **_kwargs):
        assert hostname == "example.com"
        return [(None, None, None, None, ("93.184.216.34", 0))]

    def fake_fetch_url_content(url, _settings):
        assert url == "https://example.com/policy"
        return ParsedContent(
            title="URL Policy",
            text="URL policy documents explain that annual refunds are available within 30 calendar days.",
            source_type="url",
            source_uri=url,
            metadata={"category": "policy"},
        )

    monkeypatch.setattr(url_loader.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(document_service_module, "fetch_url_content", fake_fetch_url_content)

    response = client.post("/api/v1/documents/urls", json={"url": "https://example.com/policy"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["payload"]["source"] == "urls"
    assert run_once(client.app.state.runtime) is True

    completed = client.get(f"/api/v1/jobs/{payload['id']}").json()
    assert completed["status"] == "completed"
    assert completed["result"]["documents"][0]["document_id"] == "url-policy"

    docs_response = client.get("/api/v1/documents")
    assert docs_response.status_code == 200
    documents = docs_response.json()["documents"]
    assert any(item["source_type"] == "url" and item["source_uri"] == "https://example.com/policy" for item in documents)


def test_eval_datasets_are_listed(client):
    response = client.get("/api/v1/evals/datasets")
    assert response.status_code == 200
    datasets = response.json()["datasets"]
    assert any(item["name"] == "portfolio_eval.jsonl" and item["status"] == "valid" for item in datasets)


def test_eval_dataset_name_rejects_paths(client):
    response = client.post("/api/v1/evals/runs", json={"dataset_name": "../portfolio_eval.jsonl"})

    assert response.status_code == 422
