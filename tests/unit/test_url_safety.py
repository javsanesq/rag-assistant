import pytest

from rag_assistant_api.adapters.url_loader import expand_urls
from rag_assistant_api.core.config import Settings


def test_private_url_blocked_by_default():
    settings = Settings(embed_provider="mock", llm_provider="mock")
    with pytest.raises(ValueError, match="blocked"):
        expand_urls("http://127.0.0.1:8000", [], None, settings)


def test_invalid_scheme_rejected():
    settings = Settings(embed_provider="mock", llm_provider="mock")
    with pytest.raises(ValueError, match="http"):
        expand_urls("file:///etc/passwd", [], None, settings)
