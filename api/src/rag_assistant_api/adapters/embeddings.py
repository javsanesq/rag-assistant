from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

from openai import OpenAI

from rag_assistant_api.core.config import Settings
from rag_assistant_api.core.exceptions import ProviderConfigurationError


class EmbeddingProvider(ABC):
    dimensions: int

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class MockEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = []
        for index in range(self.dimensions):
            byte = digest[index % len(digest)]
            vector.append((byte / 255.0) * 2 - 1)
        return vector


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ProviderConfigurationError(
                "sentence-transformers support is optional. Install with `pip install -e api[local-embeddings]` "
                "or use EMBED_PROVIDER=openai/mock."
            ) from exc

        self.model = SentenceTransformer(model_name)
        self.dimensions = int(self.model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ProviderConfigurationError("OPENAI_API_KEY is required for OpenAI embeddings.")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.embed_model
        self.dimensions = _guess_dimensions(settings.embed_model, settings.qdrant_vector_size)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    provider = settings.embed_provider.lower()
    if provider == "mock":
        return MockEmbeddingProvider(settings.qdrant_vector_size)
    if provider == "sentence-transformers":
        return SentenceTransformerEmbeddingProvider(settings.embed_model)
    if provider == "openai":
        return OpenAIEmbeddingProvider(settings)
    raise ProviderConfigurationError(f"Unsupported embedding provider: {settings.embed_provider}")


def _guess_dimensions(model_name: str, fallback: int) -> int:
    if "text-embedding-3-large" in model_name:
        return 3072
    if "text-embedding-3-small" in model_name:
        return 1536
    return fallback
