from __future__ import annotations

from datetime import date, datetime
from typing import Any

from rag_assistant_api.adapters.parsers import parse_document_date


def merge_metadata(manual: dict[str, Any], extracted: dict[str, Any]) -> dict[str, Any]:
    merged = dict(extracted)
    for key, value in manual.items():
        if value not in (None, "", []):
            merged[key] = value
    return merged


def normalize_document_date(metadata: dict[str, Any]) -> date | None:
    for key in ("document_date", "date", "published_at"):
        if key in metadata:
            return parse_document_date(metadata[key])
    return None


def to_timestamp(value: date | None) -> int | None:
    if value is None:
        return None
    return int(datetime.combine(value, datetime.min.time()).timestamp())
