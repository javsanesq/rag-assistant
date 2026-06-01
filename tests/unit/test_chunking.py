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


def test_chunk_text_enforces_size_for_single_large_paragraph():
    text = " ".join(f"token{index}" for index in range(300))

    chunks = chunk_text(text, ChunkingConfig(chunk_size=100, chunk_overlap=10))

    assert len(chunks) >= 3
    assert all(len(chunk.text.split()) <= 100 for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_chunk_text_does_not_emit_overlap_only_tail_chunk():
    text = " ".join(f"token{index}" for index in range(100))

    chunks = chunk_text(text, ChunkingConfig(chunk_size=100, chunk_overlap=20))

    assert len(chunks) == 1
    assert len(chunks[0].text.split()) == 100


def test_chunk_ids_are_deterministic_for_repeatable_evals():
    config = ChunkingConfig(chunker_type="recursive", chunk_size=4, chunk_overlap=1)

    first = chunk_text("one two three four five", config)
    second = chunk_text("one two three four five", config)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
