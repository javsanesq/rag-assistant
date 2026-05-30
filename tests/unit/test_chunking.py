from rag_assistant_api.domain.schemas import ChunkingConfig
from rag_assistant_api.services.chunking import chunk_text, validate_chunking


def test_validate_chunking_caps_size():
    config = ChunkingConfig(chunker_type="recursive", chunk_size=5000, chunk_overlap=900)
    validated = validate_chunking(config, max_chunk_size=1000)
    assert validated.chunk_size == 1000
    assert validated.chunk_overlap <= 500


def test_chunk_text_returns_multiple_chunks():
    text = "\n".join([f"Paragraph {index} " + ("word " * 100) for index in range(8)])
    chunks = chunk_text(text, ChunkingConfig(chunk_size=120, chunk_overlap=20))
    assert len(chunks) >= 2
    assert chunks[0].chunk_index == 0
