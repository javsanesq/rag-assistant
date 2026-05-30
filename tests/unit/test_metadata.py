from rag_assistant_api.services.metadata import merge_metadata, normalize_document_date


def test_manual_metadata_wins():
    merged = merge_metadata(
        {"category": "policy", "title": "Manual Title"},
        {"category": "engineering", "title": "Extracted Title"},
    )
    assert merged["category"] == "policy"
    assert merged["title"] == "Manual Title"


def test_normalize_document_date_reads_common_keys():
    metadata = {"published_at": "2026-04-01"}
    assert normalize_document_date(metadata).isoformat() == "2026-04-01"
