from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from uuid import NAMESPACE_URL, uuid5

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
    chunk_size = max(1, config.chunk_size)
    chunk_overlap = min(max(0, config.chunk_overlap), max(0, chunk_size - 1))
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
    has_new_words = False

    def flush_chunk() -> None:
        nonlocal index, words, has_new_words
        if not words or not has_new_words:
            return
        chunk_text = " ".join(words)
        chunks.append(
            TextChunk(
                chunk_id=str(uuid5(NAMESPACE_URL, f"rag-assistant:{index}:{chunk_text}")),
                chunk_index=index,
                text=chunk_text,
            )
        )
        index += 1
        words = words[-chunk_overlap:] if chunk_overlap else []
        has_new_words = False

    for block in blocks:
        block_words = block.split()
        while block_words:
            available = chunk_size - len(words)
            if available <= 0:
                flush_chunk()
                available = chunk_size - len(words)
            take, block_words = block_words[:available], block_words[available:]
            words.extend(take)
            has_new_words = has_new_words or bool(take)
            if len(words) >= chunk_size:
                flush_chunk()
    flush_chunk()
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
