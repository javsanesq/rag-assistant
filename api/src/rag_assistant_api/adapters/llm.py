from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

import httpx
from openai import OpenAI

from rag_assistant_api.core.config import Settings
from rag_assistant_api.core.exceptions import ProviderConfigurationError


class LLMProvider(ABC):
    @abstractmethod
    def answer(self, question: str, context_blocks: list[str]) -> str:
        raise NotImplementedError

    @abstractmethod
    def judge_faithfulness(self, question: str, answer: str, citations: list[dict]) -> dict:
        raise NotImplementedError


class MockLLMProvider(LLMProvider):
    def answer(self, question: str, context_blocks: list[str]) -> str:
        if not context_blocks:
            return "I could not find grounded context for that question in the indexed documents."
        primary = context_blocks[0]
        match = re.match(r"^\[(\d+)\]\s*(.*)$", primary, flags=re.DOTALL)
        if match:
            content = match.group(2).strip()
            lines = content.splitlines()
            if lines and lines[0].startswith("chunk_id="):
                content = "\n".join(lines[1:]).strip()
            return f"Grounded answer: {content[:450].strip()} [{match.group(1)}]"
        return f"Grounded answer: {primary[:450].strip()}"

    def judge_faithfulness(self, question: str, answer: str, citations: list[dict]) -> dict:
        if not citations:
            return {"score": 1, "rationale": "No citations were retrieved, so the answer is not verifiable."}
        overlap = 0
        answer_tokens = set(answer.lower().split())
        citation_tokens = set(" ".join(item["excerpt"].lower() for item in citations).split())
        if answer_tokens:
            overlap = int((len(answer_tokens & citation_tokens) / len(answer_tokens)) * 5)
        score = min(5, max(1, overlap or 2))
        rationale = "Heuristic rubric based on answer overlap with cited excerpts."
        return {"score": score, "rationale": rationale}


class OpenAILLMProvider(LLMProvider):
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ProviderConfigurationError("OPENAI_API_KEY is required for the OpenAI LLM provider.")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model

    def answer(self, question: str, context_blocks: list[str]) -> str:
        prompt = (
            "You are a grounded RAG assistant. Answer only with information supported by the provided context. "
            "Every factual sentence must include at least one citation marker from the provided context, such as [1]. "
            "Use only citation markers that appear in the context. If the context is insufficient, say: "
            "\"I do not have enough evidence in the indexed documents to answer that.\" and cite nothing.\n\n"
            f"Question: {question}\n\n"
            "Context:\n"
            + "\n\n".join(context_blocks)
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    def judge_faithfulness(self, question: str, answer: str, citations: list[dict]) -> dict:
        prompt = (
            "Score the answer on a 1-5 faithfulness rubric where 5 means fully supported by citations and "
            "1 means unsupported or contradictory. Return JSON with keys score and rationale.\n\n"
            f"Question: {question}\nAnswer: {answer}\nCitations: {json.dumps(citations)}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"score": 3, "rationale": content.strip()}


class OllamaLLMProvider(LLMProvider):
    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model

    def answer(self, question: str, context_blocks: list[str]) -> str:
        prompt = (
            "Answer only from the context below. Every factual sentence must include a citation marker like [1]. "
            "Use only citation markers from the context. If the context is insufficient, say that directly without citations.\n\n"
            f"Question: {question}\n\nContext:\n" + "\n\n".join(context_blocks)
        )
        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json().get("response", "")

    def judge_faithfulness(self, question: str, answer: str, citations: list[dict]) -> dict:
        prompt = (
            "Return JSON {\"score\": 1-5, \"rationale\": \"...\"} for the faithfulness of the answer to the citations.\n"
            f"Question: {question}\nAnswer: {answer}\nCitations: {json.dumps(citations)}"
        )
        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=60.0,
        )
        response.raise_for_status()
        content = response.json().get("response", "{}")
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"score": 3, "rationale": content.strip()}


def build_llm_provider(settings: Settings) -> LLMProvider:
    provider = settings.llm_provider.lower()
    if provider == "mock":
        return MockLLMProvider()
    if provider == "openai":
        return OpenAILLMProvider(settings)
    if provider == "ollama":
        return OllamaLLMProvider(settings)
    raise ProviderConfigurationError(f"Unsupported LLM provider: {settings.llm_provider}")
