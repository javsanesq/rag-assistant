import pytest

from rag_assistant_api.adapters import url_loader
from rag_assistant_api.adapters.url_loader import (
    MAX_URL_REDIRECTS,
    expand_urls,
    fetch_url_content,
)
from rag_assistant_api.core.config import Settings


class FakeStreamResponse:
    def __init__(
        self,
        url: str,
        status_code: int,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
    ):
        self.url = url
        self.status_code = status_code
        self.headers = headers or {}
        self.body = body
        self.encoding = "utf-8"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_bytes(self):
        yield self.body


def test_private_url_blocked_by_default():
    settings = Settings(embed_provider="mock", llm_provider="mock")
    with pytest.raises(ValueError, match="blocked"):
        expand_urls("http://127.0.0.1:8000", [], None, settings)


def test_invalid_scheme_rejected():
    settings = Settings(embed_provider="mock", llm_provider="mock")
    with pytest.raises(ValueError, match="http"):
        expand_urls("file:///etc/passwd", [], None, settings)


def test_redirect_to_local_url_blocked(monkeypatch):
    settings = Settings(embed_provider="mock", llm_provider="mock")

    def fake_getaddrinfo(hostname, *_args, **_kwargs):
        if hostname == "example.com":
            return [(None, None, None, None, ("93.184.216.34", 0))]
        if hostname == "127.0.0.1":
            return [(None, None, None, None, ("127.0.0.1", 0))]
        raise AssertionError(f"Unexpected host lookup: {hostname}")

    def fake_stream(method, url, **_kwargs):
        assert method == "GET"
        return FakeStreamResponse(url, 302, {"location": "http://127.0.0.1:8000/admin"})

    monkeypatch.setattr(url_loader.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(url_loader.httpx, "stream", fake_stream)

    with pytest.raises(ValueError, match="Private or local URL targets are blocked"):
        fetch_url_content("https://example.com/page", settings)


def test_redirect_to_private_url_blocked(monkeypatch):
    settings = Settings(embed_provider="mock", llm_provider="mock")

    def fake_getaddrinfo(hostname, *_args, **_kwargs):
        if hostname == "example.com":
            return [(None, None, None, None, ("93.184.216.34", 0))]
        if hostname == "metadata.internal":
            return [(None, None, None, None, ("169.254.169.254", 0))]
        raise AssertionError(f"Unexpected host lookup: {hostname}")

    def fake_stream(method, url, **_kwargs):
        assert method == "GET"
        return FakeStreamResponse(url, 301, {"location": "http://metadata.internal/latest/meta-data"})

    monkeypatch.setattr(url_loader.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(url_loader.httpx, "stream", fake_stream)

    with pytest.raises(ValueError, match="Private or local URL targets are blocked"):
        fetch_url_content("https://example.com/page", settings)


def test_redirect_limit_enforced(monkeypatch):
    settings = Settings(embed_provider="mock", llm_provider="mock")
    requested_urls = []

    def fake_getaddrinfo(hostname, *_args, **_kwargs):
        if hostname == "example.com":
            return [(None, None, None, None, ("93.184.216.34", 0))]
        raise AssertionError(f"Unexpected host lookup: {hostname}")

    def fake_stream(method, url, **_kwargs):
        assert method == "GET"
        requested_urls.append(url)
        next_hop = int(url.rsplit("/", 1)[-1]) + 1
        return FakeStreamResponse(url, 302, {"location": f"/{next_hop}"})

    monkeypatch.setattr(url_loader.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(url_loader.httpx, "stream", fake_stream)

    with pytest.raises(ValueError, match="URL redirect limit exceeded"):
        fetch_url_content("https://example.com/0", settings)

    assert len(requested_urls) == MAX_URL_REDIRECTS + 1
