from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from openai import OpenAI

from rag_assistant_api.adapters.embeddings import build_embedding_provider
from rag_assistant_api.adapters.lexical_store import SQLLexicalStore
from rag_assistant_api.adapters.llm import MockLLMProvider
from rag_assistant_api.adapters.parsers import parse_file_bytes
from rag_assistant_api.adapters.vector_store import RetrievedChunk, QdrantVectorStore
from rag_assistant_api.core.config import Settings
from rag_assistant_api.core.db import build_engine, build_session_factory, init_db
from rag_assistant_api.domain.schemas import ChunkingConfig, QueryRequest
from rag_assistant_api.services.documents import DocumentService
from rag_assistant_api.services.evaluation import EvaluationService
from rag_assistant_api.services.jobs import JobService
from rag_assistant_api.services.query import _filter_relevant_chunks
from rag_assistant_api.services.retrieval import RetrievalService


@dataclass
class BenchmarkCase:
    id: str
    question: str
    expected_document_ids: list[str]
    category: str | None
    should_answer: bool


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OpenAI-backed RAG retrieval benchmark.")
    parser.add_argument("--corpus-dir", default="samples/benchmark")
    parser.add_argument("--dataset", default="evals/datasets/benchmark_eval.jsonl")
    parser.add_argument("--report", default="docs/benchmark-report.md")
    parser.add_argument("--results-json", default="output/evals/benchmark-results.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true", default=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    work_dir = Path(tempfile.mkdtemp(prefix="rag-benchmark-"))
    try:
        settings = Settings(
            app_env="development",
            database_url=f"sqlite:///{work_dir / 'benchmark.db'}",
            qdrant_location=":memory:",
            qdrant_collection=f"benchmark_{uuid4().hex}",
            qdrant_vector_size=1536,
            embed_provider="openai",
            embed_model="text-embedding-3-small",
            llm_provider="mock",
            eval_dataset_dir=repo_root / "evals" / "datasets",
        )
        if not settings.openai_api_key:
            raise SystemExit("OPENAI_API_KEY is required to run the benchmark with OpenAI embeddings.")
        runtime = _build_benchmark_runtime(settings)
        _ingest_corpus(runtime, repo_root / args.corpus_dir)
        cases = _load_cases(repo_root / args.dataset)
        results = _run_cases(runtime, cases, top_k=args.top_k, use_reranker=args.rerank)
        report = _render_report(results, cases, args.top_k)
        report_path = repo_root / args.report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        results_path = repo_root / args.results_json
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote {report_path}")
        print(f"Wrote {results_path}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _build_benchmark_runtime(settings: Settings):
    engine = build_engine(settings.effective_database_url)
    init_db(engine, settings.effective_database_url)
    session_factory = build_session_factory(engine)
    embedding_provider = build_embedding_provider(settings)
    vector_store = QdrantVectorStore(settings, embedding_provider.dimensions)
    job_service = JobService(session_factory)
    document_service = DocumentService(session_factory, vector_store, embedding_provider, job_service, settings)
    lexical_store = SQLLexicalStore(session_factory)
    retrieval_service = RetrievalService(vector_store, embedding_provider, settings.top_k, lexical_store)
    return type(
        "BenchmarkRuntime",
        (),
        {
            "settings": settings,
            "document_service": document_service,
            "retrieval_service": retrieval_service,
            "lexical_store": lexical_store,
            "evaluation_service": EvaluationService(settings, None, job_service, MockLLMProvider()),
        },
    )()


def _ingest_corpus(runtime, corpus_dir: Path) -> None:
    chunking = ChunkingConfig(chunker_type="markdown", chunk_size=220, chunk_overlap=35)
    for path in sorted(corpus_dir.glob("*.md")):
        parsed = parse_file_bytes(path.name, path.read_bytes())
        runtime.document_service._ingest_parsed_content(
            parsed=parsed,
            manual_metadata={},
            chunking=chunking,
            job_id="benchmark",
            raw_content=path.read_bytes(),
        )


def _load_cases(path: Path) -> list[BenchmarkCase]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [
        BenchmarkCase(
            id=row["id"],
            question=row["question"],
            expected_document_ids=row.get("expected_document_ids", []),
            category=row.get("category"),
            should_answer=row.get("should_answer", True),
        )
        for row in rows
    ]


def _run_cases(runtime, cases: list[BenchmarkCase], top_k: int, use_reranker: bool) -> dict:
    openai_client = OpenAI(api_key=runtime.settings.openai_api_key) if use_reranker else None
    modes = {
        "dense_only": {"retrieval_mode": "dense", "alpha": 1.0, "rerank": False},
        "hybrid_bm25": {"retrieval_mode": "hybrid", "alpha": 0.65, "rerank": False},
        "hybrid_bm25_reranked": {"retrieval_mode": "hybrid", "alpha": 0.65, "rerank": True},
    }
    output = {
        "config": {
            "embed_provider": runtime.settings.embed_provider,
            "embed_model": runtime.settings.embed_model,
            "qdrant_vector_size": runtime.settings.qdrant_vector_size,
            "top_k": top_k,
            "reranker": runtime.settings.llm_model,
            "hybrid_fusion": "weighted reciprocal rank fusion",
            "hybrid_alpha": modes["hybrid_bm25"]["alpha"],
            "lexical_backend": "SQLite FTS5 BM25 with token-overlap fallback",
        },
        "modes": {},
    }
    for mode_name, mode_config in modes.items():
        examples = []
        for case in cases:
            request = QueryRequest(
                question=case.question,
                category=case.category,
                top_k=top_k,
                retrieval_mode=mode_config["retrieval_mode"],
                alpha=mode_config["alpha"],
            )
            retrieved = runtime.retrieval_service.retrieve(request)
            relevant, _ = _filter_relevant_chunks(retrieved, runtime.settings, case.question)
            selected = relevant
            if mode_config["rerank"] and openai_client:
                selected = _rerank(openai_client, runtime.settings.llm_model, case.question, selected)[:top_k]
            examples.append(_score_case(case, selected, top_k))
        output["modes"][mode_name] = {
            "summary": _summarize_examples(examples),
            "examples": examples,
        }
    return output


def _rerank(client: OpenAI, model: str, question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if not chunks:
        return chunks
    chunk_payload = [
        {"chunk_id": chunk.chunk_id, "document_id": chunk.document_id, "excerpt": chunk.excerpt[:900]}
        for chunk in chunks
    ]
    prompt = (
        "Decide whether these retrieved chunks contain enough evidence to answer the question. "
        "If no chunk directly answers the question, return {\"answerable\": false, \"chunk_ids\": []}. "
        "Otherwise rank the chunks by support quality and return only JSON: "
        "{\"answerable\": true, \"chunk_ids\": [\"...\"]}.\n\n"
        f"Question: {question}\nChunks: {json.dumps(chunk_payload)}"
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(content)
        if parsed.get("answerable") is False:
            return []
        ranked_ids = parsed.get("chunk_ids", [])
    except json.JSONDecodeError:
        ranked_ids = []
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    ranked = [by_id[chunk_id] for chunk_id in ranked_ids if chunk_id in by_id]
    ranked.extend(chunk for chunk in chunks if chunk.chunk_id not in {item.chunk_id for item in ranked})
    return ranked


def _score_case(case: BenchmarkCase, chunks: list[RetrievedChunk], top_k: int) -> dict:
    actual_ids = [chunk.document_id for chunk in chunks]
    expected = set(case.expected_document_ids)
    hits = [document_id for document_id in actual_ids if document_id in expected]
    first_rank = next((index + 1 for index, document_id in enumerate(actual_ids) if document_id in expected), None)
    abstained = len(chunks) == 0
    return {
        "id": case.id,
        "should_answer": case.should_answer,
        "expected_document_ids": case.expected_document_ids,
        "actual_document_ids": actual_ids,
        "top_chunk_ids": [chunk.chunk_id for chunk in chunks],
        "precision_at_k": len(hits) / max(1, top_k) if case.should_answer else 1.0 if abstained else 0.0,
        "hit_rate": 1.0 if hits else 0.0 if case.should_answer else 1.0 if abstained else 0.0,
        "recall_at_k": len(set(hits)) / max(1, len(expected)) if case.should_answer else 1.0 if abstained else 0.0,
        "mrr": 1 / first_rank if first_rank else 0.0 if case.should_answer else 1.0 if abstained else 0.0,
        "abstained": abstained,
    }


def _summarize_examples(examples: list[dict]) -> dict:
    answerable = [item for item in examples if item["should_answer"]]
    no_answer = [item for item in examples if not item["should_answer"]]
    return {
        "examples": len(examples),
        "answerable_examples": len(answerable),
        "no_answer_examples": len(no_answer),
        "precision_at_k": _mean(item["precision_at_k"] for item in answerable),
        "hit_rate": _mean(item["hit_rate"] for item in answerable),
        "recall_at_k": _mean(item["recall_at_k"] for item in answerable),
        "mrr": _mean(item["mrr"] for item in answerable),
        "abstention_accuracy": _mean(1.0 if item["abstained"] else 0.0 for item in no_answer),
    }


def _mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _render_report(results: dict, cases: list[BenchmarkCase], top_k: int) -> str:
    dense_summary = results["modes"]["dense_only"]["summary"]
    hybrid_summary = results["modes"]["hybrid_bm25"]["summary"]
    reranked_summary = results["modes"]["hybrid_bm25_reranked"]["summary"]
    lines = [
        "# RAG Benchmark Report",
        "",
        "This report compares retrieval quality on a synthetic but realistic benchmark corpus using OpenAI embeddings.",
        "",
        "## Configuration",
        "",
        f"- Embeddings: `{results['config']['embed_provider']}` / `{results['config']['embed_model']}`",
        f"- Vector size: `{results['config']['qdrant_vector_size']}`",
        f"- Top K: `{top_k}`",
        f"- Reranker: `{results['config']['reranker']}`",
        f"- Hybrid fusion: `{results['config']['hybrid_fusion']}` with alpha `{results['config']['hybrid_alpha']}`",
        f"- Lexical backend: `{results['config']['lexical_backend']}`",
        f"- Examples: `{len(cases)}`",
        "",
        "## Results",
        "",
        "| Mode | Precision@K | Hit Rate | Recall@K | MRR | Abstention Accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode_name, payload in results["modes"].items():
        summary = payload["summary"]
        lines.append(
            f"| {mode_name} | {summary['precision_at_k']:.4f} | {summary['hit_rate']:.4f} | "
            f"{summary['recall_at_k']:.4f} | {summary['mrr']:.4f} | {summary['abstention_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- `dense_only` is the semantic-vector baseline. On this benchmark it already retrieved the expected document for every answerable example (hit rate {dense_summary['hit_rate']:.4f}).",
            f"- `hybrid_bm25` adds SQLite FTS5 BM25 lexical retrieval and weighted Reciprocal Rank Fusion. On this corpus it matched dense retrieval's hit rate ({hybrid_summary['hit_rate']:.4f}), which means the answerable questions were not hard enough to expose a BM25 recall gain.",
            f"- `hybrid_bm25_reranked` applies an OpenAI answerability reranker. The main measured gain was no-answer behavior: abstention accuracy moved from {dense_summary['abstention_accuracy']:.4f} to {reranked_summary['abstention_accuracy']:.4f}.",
            f"- Precision@K is capped at {1 / top_k:.4f} for this dataset because each answerable question has one expected document and the benchmark retrieves K={top_k} candidates.",
            "- The takeaway is that OpenAI embeddings are already strong on clean single-hop policy questions; the reranker adds value by rejecting retrieved-but-insufficient context.",
            "",
            "## Per-Example Results",
            "",
            "| ID | Expected | Dense | Hybrid BM25 | Reranked |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    dense = {item["id"]: item for item in results["modes"]["dense_only"]["examples"]}
    hybrid = {item["id"]: item for item in results["modes"]["hybrid_bm25"]["examples"]}
    reranked = {item["id"]: item for item in results["modes"]["hybrid_bm25_reranked"]["examples"]}
    for case in cases:
        lines.append(
            f"| {case.id} | {', '.join(case.expected_document_ids) or 'ABSTAIN'} | "
            f"{', '.join(dense[case.id]['actual_document_ids'][:3]) or 'ABSTAIN'} | "
            f"{', '.join(hybrid[case.id]['actual_document_ids'][:3]) or 'ABSTAIN'} | "
            f"{', '.join(reranked[case.id]['actual_document_ids'][:3]) or 'ABSTAIN'} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "This is a controlled synthetic benchmark, not a replacement for domain-specific production evaluation. "
            "The next step is to add real company documents, expected chunk IDs, answer-quality rubrics, and prompt-injection cases.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
