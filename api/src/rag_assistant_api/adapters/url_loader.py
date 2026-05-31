from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

import httpx
from bs4 import BeautifulSoup

from rag_assistant_api.adapters.parsers import ParsedContent
from rag_assistant_api.core.config import Settings

MAX_URL_REDIRECTS = 5
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


def expand_urls(url: str | None, urls: list[str], sitemap_url: str | None, settings: Settings) -> list[str]:
    expanded = list(urls)
    if url:
        expanded.append(url)
    if sitemap_url:
        _validate_url(sitemap_url, settings)
        expanded.extend(_fetch_sitemap_urls(sitemap_url, settings))
    deduped: list[str] = []
    for item in expanded:
        _validate_url(item, settings)
        if item not in deduped:
            deduped.append(item)
        if len(deduped) > settings.max_sitemap_urls:
            raise ValueError(f"URL ingestion is limited to {settings.max_sitemap_urls} URLs.")
    return deduped


def fetch_url_content(url: str, settings: Settings) -> ParsedContent:
    _validate_url(url, settings)
    response_text = _bounded_get(url, settings)
    soup = BeautifulSoup(response_text, "html.parser")
    title = (soup.title.string or url) if soup.title else url
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    text = " ".join(chunk.strip() for chunk in soup.stripped_strings)
    return ParsedContent(
        title=title.strip(),
        text=text.strip(),
        source_type="url",
        source_uri=url,
        metadata={},
    )


def _fetch_sitemap_urls(sitemap_url: str, settings: Settings) -> list[str]:
    root = ElementTree.fromstring(_bounded_get(sitemap_url, settings))
    namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [node.text for node in root.findall(".//sm:loc", namespace) if node.text]
    return urls[: settings.max_sitemap_urls]


def _bounded_get(url: str, settings: Settings) -> str:
    current_url = url
    redirects_followed = 0

    while True:
        _validate_url(current_url, settings)
        with httpx.stream("GET", current_url, timeout=20.0, follow_redirects=False) as response:
            if response.status_code in _REDIRECT_STATUSES:
                if redirects_followed >= MAX_URL_REDIRECTS:
                    raise ValueError(f"URL redirect limit exceeded: {MAX_URL_REDIRECTS}")
                location = response.headers.get("location")
                if not location:
                    raise ValueError("URL redirect response is missing a Location header.")
                next_url = urljoin(str(response.url), location)
                _validate_url(next_url, settings)
                current_url = next_url
                redirects_followed += 1
                continue

            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if not any(item in content_type for item in ("text/html", "text/xml", "application/xml", "application/xhtml+xml")):
                raise ValueError(f"Unsupported URL content type: {content_type or 'unknown'}")
            chunks = []
            total = 0
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > settings.max_url_bytes:
                    raise ValueError("URL response exceeds MAX_URL_BYTES.")
                chunks.append(chunk)
            return b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")


def _validate_url(url: str, settings: Settings) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only absolute http(s) URLs are supported.")
    hostname = parsed.hostname.lower()
    if settings.url_allowed_domains and hostname not in settings.url_allowed_domains:
        raise ValueError(f"URL host is not in URL_ALLOWED_DOMAINS: {hostname}")
    if hostname in settings.url_blocked_domains:
        raise ValueError(f"URL host is blocked: {hostname}")
    if settings.allow_private_urls:
        return
    try:
        addresses = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve URL host: {hostname}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError(f"Private or local URL targets are blocked: {hostname}")
