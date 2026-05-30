from __future__ import annotations

from xml.etree import ElementTree

import httpx
from bs4 import BeautifulSoup

from rag_assistant_api.adapters.parsers import ParsedContent


def expand_urls(url: str | None, urls: list[str], sitemap_url: str | None) -> list[str]:
    expanded = list(urls)
    if url:
        expanded.append(url)
    if sitemap_url:
        expanded.extend(_fetch_sitemap_urls(sitemap_url))
    deduped: list[str] = []
    for item in expanded:
        if item not in deduped:
            deduped.append(item)
    return deduped


def fetch_url_content(url: str) -> ParsedContent:
    response = httpx.get(url, timeout=20.0, follow_redirects=True)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
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


def _fetch_sitemap_urls(sitemap_url: str) -> list[str]:
    response = httpx.get(sitemap_url, timeout=20.0, follow_redirects=True)
    response.raise_for_status()
    root = ElementTree.fromstring(response.text)
    namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [node.text for node in root.findall(".//sm:loc", namespace) if node.text]
    return urls
