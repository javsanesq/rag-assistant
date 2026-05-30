from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any

import yaml
from docx import Document as DocxDocument
from pypdf import PdfReader

from rag_assistant_api.core.exceptions import UnsupportedSourceError


@dataclass
class ParsedContent:
    title: str
    text: str
    source_type: str
    source_uri: str
    metadata: dict[str, Any]


def parse_document_date(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def parse_file_bytes(filename: str, content: bytes) -> ParsedContent:
    suffix = Path(filename).suffix.lower()
    if suffix == ".md":
        return _parse_markdown(filename, content.decode("utf-8"))
    if suffix == ".pdf":
        reader = PdfReader(BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return ParsedContent(
            title=Path(filename).stem.replace("-", " ").title(),
            text=text.strip(),
            source_type="pdf",
            source_uri=filename,
            metadata={},
        )
    if suffix == ".docx":
        document = DocxDocument(BytesIO(content))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text)
        return ParsedContent(
            title=Path(filename).stem.replace("-", " ").title(),
            text=text.strip(),
            source_type="docx",
            source_uri=filename,
            metadata={},
        )
    raise UnsupportedSourceError(f"Unsupported file type for {filename}")


def _parse_markdown(filename: str, raw_text: str) -> ParsedContent:
    metadata: dict[str, Any] = {}
    body = raw_text
    if raw_text.startswith("---"):
        match = re.match(r"^---\n(.*?)\n---\n(.*)$", raw_text, re.DOTALL)
        if match:
            metadata = yaml.safe_load(match.group(1)) or {}
            body = match.group(2)
    title = str(metadata.get("title") or Path(filename).stem.replace("-", " ").title())
    return ParsedContent(
        title=title,
        text=body.strip(),
        source_type="markdown",
        source_uri=filename,
        metadata=metadata,
    )
