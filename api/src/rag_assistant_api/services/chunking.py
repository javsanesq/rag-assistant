from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from uuid import uuid4

from rag_assistant_api.domain.schemas import ChunkingConfig


@dataclass
class TextChunk:
    chunk_id: str
    chunk_index: int
    text: str


def validate_chunking(config: ChunkingConfig, max_chunk_size: int) -> ChunkingConfig:
    chunk_size = min(config.chunk_size, max_chunk_size)
    chunk_overlap = min(config.chunk_overlap, max(0, chunk_size // 2))
    return ChunkingConfig(
        chunker_type=config.chunker_type,
        chunk_size=max(150, chunk_size),
        chunk_overlap=max(0, chunk_overlap),
    )


def chunk_text(text: str, config: ChunkingConfig) -> list[TextChunk]:
    normalized = text.strip()
    if not normalized:
        return []
    if config.chunker_type == "markdown":
        sections = _split_markdown_sections(normalized)
    elif config.chunker_type == "sentence":
        sections = normalized.replace("! ", ". ").replace("? ", ". ").split(". ")
    else:
        sections = normalized.splitlines()
    blocks = [block.strip() for block in sections if block.strip()]
    words = []
    chunks: list[TextChunk] = []
    index = 0
    for block in blocks:
        block_words = block.split()
        if not block_words:
            continue
        if len(words) + len(block_words) > config.chunk_size and words:
            chunks.append(TextChunk(chunk_id=str(uuid4()), chunk_index=index, text=" ".join(words)))
            index += 1
            words = words[-config.chunk_overlap :] if config.chunk_overlap else []
        words.extend(block_words)
    if words:
        chunks.append(TextChunk(chunk_id=str(uuid4()), chunk_index=index, text=" ".join(words)))
    return chunks


def _split_markdown_sections(text: str) -> Iterable[str]:
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("#") and current:
            yield "\n".join(current)
            current = [line]
        else:
            current.append(line)
    if current:
        yield "\n".join(current)
