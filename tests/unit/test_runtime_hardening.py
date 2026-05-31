import pytest

from rag_assistant_api.adapters.parsers import parse_file_bytes
from rag_assistant_api.core.config import Settings


def test_markdown_extension_is_parsed_like_md():
    parsed = parse_file_bytes("guide.markdown", b"---\ntitle: Guide\n---\n# Heading\nBody")

    assert parsed.source_type == "markdown"
    assert parsed.title == "Guide"
    assert "Body" in parsed.text


def test_mock_providers_are_rejected_in_production():
    with pytest.raises(ValueError, match="Mock providers"):
        Settings(app_env="production", embed_provider="mock", llm_provider="mock")


def test_api_auth_token_is_required_in_production():
    with pytest.raises(ValueError, match="API_AUTH_TOKEN"):
        Settings(
            app_env="production",
            embed_provider="openai",
            llm_provider="openai",
            openai_api_key="test-key",
            api_auth_token="",
        )
