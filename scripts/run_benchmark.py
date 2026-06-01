from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from rag_assistant_api.adapters.embeddings import build_embedding_provider
from rag_assistant_api.adapters.lexical_store import SQLLexicalStore
from rag_assistant_api.adapters.llm import MockLLMProvider
from rag_assistant_api.adapters.parsers import parse_file_bytes
from rag_assistant_api.adapters.reranker import OpenAIRerankerProvider
from rag_assistant_api.adapters.vector_store import RetrievedChunk, QdrantVectorStore
from rag_assistant_api.core.config import Settings
from rag_assistant_api.core.db import build_engine, build_session_factory, init_db
from rag_assistant_api.domain.schemas import ChunkingConfig, QueryRequest
from rag_assistant_api.services.documents import DocumentService
from rag_assistant_api.services.eval_metrics import answer_contains_expected, mean, score_ranked_ids
from rag_assistant_api.services.evaluation import EvaluationService
from rag_assistant_api.services.jobs import JobService
from rag_assistant_api.services.query import _filter_relevant_chunks
from rag_assistant_api.services.retrieval import RetrievalService


@dataclass
class BenchmarkCase:
    id: str
    question: str
    expected_document_ids: list[str]
    expected_chunk_ids: list[str]
    expected_answer_contains: list[str]
    document_ids: list[str]
    category: str | None
    date_from: str | None
    date_to: str | None
    should_answer: bool
    tags: list[str]
    difficulty: str | None


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
            reranker_provider="openai",
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
            expected_chunk_ids=row.get("expected_chunk_ids", []),
            expected_answer_contains=_as_list(row.get("expected_answer_contains", [])),
            document_ids=row.get("document_ids", []),
            category=row.get("category"),
            date_from=row.get("date_from"),
            date_to=row.get("date_to"),
            should_answer=row.get("should_answer", True),
            tags=row.get("tags", []) or ["untagged"],
            difficulty=row.get("difficulty"),
        )
        for row in rows
    ]


def _run_cases(runtime, cases: list[BenchmarkCase], top_k: int, use_reranker: bool) -> dict:
    reranker = OpenAIRerankerProvider(runtime.settings) if use_reranker else None
    modes = {
        "dense_only": {"retrieval_mode": "dense", "alpha": 1.0, "threshold": False, "rerank": False},
        "hybrid_bm25": {"retrieval_mode": "hybrid", "alpha": 0.65, "threshold": False, "rerank": False},
        "hybrid_bm25_thresholded": {"retrieval_mode": "hybrid", "alpha": 0.65, "threshold": True, "rerank": False},
        "hybrid_bm25_reranked": {"retrieval_mode": "hybrid", "alpha": 0.65, "threshold": True, "rerank": True},
    }
    output = {
        "config": {
            "embed_provider": runtime.settings.embed_provider,
            "embed_model": runtime.settings.embed_model,
            "qdrant_vector_size": runtime.settings.qdrant_vector_size,
            "top_k": top_k,
            "reranker": runtime.settings.reranker_model,
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
                document_ids=case.document_ids,
                category=case.category,
                date_from=case.date_from,
                date_to=case.date_to,
                top_k=top_k,
                retrieval_mode=mode_config["retrieval_mode"],
                alpha=mode_config["alpha"],
            )
            retrieved = runtime.retrieval_service.retrieve(request)
            selected = retrieved
            if mode_config["threshold"]:
                selected, _ = _filter_relevant_chunks(retrieved, runtime.settings, case.question)
            if mode_config["rerank"] and reranker:
                decision = reranker.rerank(case.question, selected)
                selected = decision.selected_chunks if decision.answerable else []
            selected = selected[:top_k]
            examples.append(_score_case(case, selected, top_k))
        output["modes"][mode_name] = {
            "summary": _summarize_examples(examples),
            "examples": examples,
        }
    return output


def _score_case(case: BenchmarkCase, chunks: list[RetrievedChunk], top_k: int) -> dict:
    actual_ids = [chunk.document_id for chunk in chunks]
    actual_chunk_ids = [chunk.chunk_id for chunk in chunks]
    expected = set(case.expected_document_ids)
    hits = [document_id for document_id in actual_ids if document_id in expected]
    abstained = len(chunks) == 0
    document_scores = score_ranked_ids(case.expected_document_ids, actual_ids, top_k, case.should_answer)
    chunk_scores = (
        score_ranked_ids(case.expected_chunk_ids, actual_chunk_ids, top_k, case.should_answer)
        if case.expected_chunk_ids
        else document_scores
    )
    answer_contains = answer_contains_expected(" ".join(chunk.excerpt for chunk in chunks), case.expected_answer_contains)
    if not case.should_answer:
        answer_contains = abstained
    return {
        "id": case.id,
        "should_answer": case.should_answer,
        "tags": case.tags,
        "difficulty": case.difficulty,
        "expected_document_ids": case.expected_document_ids,
        "expected_chunk_ids": case.expected_chunk_ids,
        "actual_document_ids": actual_ids,
        "actual_chunk_ids": actual_chunk_ids,
        "top_chunk_ids": [chunk.chunk_id for chunk in chunks],
        "precision_at_k": document_scores["precision_at_k"],
        "hit_rate": document_scores["hit_rate"],
        "recall_at_k": document_scores["recall_at_k"],
        "mrr": document_scores["mrr"],
        "chunk_precision_at_k": chunk_scores["precision_at_k"],
        "chunk_hit_rate": chunk_scores["hit_rate"],
        "chunk_recall_at_k": chunk_scores["recall_at_k"],
        "chunk_mrr": chunk_scores["mrr"],
        "answer_contains": bool(answer_contains),
        "abstained": abstained,
    }


def _summarize_examples(examples: list[dict]) -> dict:
    answerable = [item for item in examples if item["should_answer"]]
    no_answer = [item for item in examples if not item["should_answer"]]
    return {
        "examples": len(examples),
        "answerable_examples": len(answerable),
        "no_answer_examples": len(no_answer),
        "precision_at_k": mean(item["precision_at_k"] for item in answerable),
        "hit_rate": mean(item["hit_rate"] for item in answerable),
        "recall_at_k": mean(item["recall_at_k"] for item in answerable),
        "mrr": mean(item["mrr"] for item in answerable),
        "chunk_precision_at_k": mean(item["chunk_precision_at_k"] for item in answerable),
        "chunk_hit_rate": mean(item["chunk_hit_rate"] for item in answerable),
        "chunk_recall_at_k": mean(item["chunk_recall_at_k"] for item in answerable),
        "chunk_mrr": mean(item["chunk_mrr"] for item in answerable),
        "answer_contains_rate": mean(1.0 if item["answer_contains"] else 0.0 for item in examples),
        "abstention_accuracy": mean(1.0 if item["abstained"] else 0.0 for item in no_answer),
        "by_tag": _summarize_by_tag(examples),
    }


def _summarize_by_tag(examples: list[dict]) -> dict:
    by_tag: dict[str, list[dict]] = {}
    for item in examples:
        for tag in item["tags"]:
            by_tag.setdefault(tag, []).append(item)
    return {
        tag: {
            "examples": len(items),
            "document_hit_rate": mean(item["hit_rate"] for item in items),
            "chunk_hit_rate": mean(item["chunk_hit_rate"] for item in items),
            "answer_contains_rate": mean(1.0 if item["answer_contains"] else 0.0 for item in items),
            "abstention_accuracy": mean(1.0 if item["abstained"] else 0.0 for item in items if not item["should_answer"]),
        }
        for tag, items in sorted(by_tag.items())
    }


def _render_report(results: dict, cases: list[BenchmarkCase], top_k: int) -> str:
    dense_summary = results["modes"]["dense_only"]["summary"]
    hybrid_summary = results["modes"]["hybrid_bm25"]["summary"]
    thresholded_summary = results["modes"]["hybrid_bm25_thresholded"]["summary"]
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
        "| Mode | Doc Hit | Chunk Hit | Chunk MRR | Answer Contains | Abstention Accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode_name, payload in results["modes"].items():
        summary = payload["summary"]
        lines.append(
            f"| {mode_name} | {summary['hit_rate']:.4f} | {summary['chunk_hit_rate']:.4f} | "
            f"{summary['chunk_mrr']:.4f} | {summary['answer_contains_rate']:.4f} | {summary['abstention_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Results By Tag",
            "",
            "| Mode | Tag | Examples | Doc Hit | Chunk Hit | Answer Contains | Abstention |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode_name, payload in results["modes"].items():
        for tag, stats in payload["summary"]["by_tag"].items():
            lines.append(
                f"| {mode_name} | {tag} | {stats['examples']} | {stats['document_hit_rate']:.4f} | "
                f"{stats['chunk_hit_rate']:.4f} | {stats['answer_contains_rate']:.4f} | "
                f"{stats['abstention_accuracy']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- `dense_only` is the semantic-vector baseline. On this benchmark it already retrieved the expected document for every answerable example (hit rate {dense_summary['hit_rate']:.4f}).",
            f"- `hybrid_bm25` adds SQLite FTS5 BM25 lexical retrieval and weighted Reciprocal Rank Fusion. On this corpus it matched dense retrieval's hit rate ({hybrid_summary['hit_rate']:.4f}), which means the answerable questions were not hard enough to expose a BM25 recall gain.",
            f"- `hybrid_bm25_thresholded` applies the production relevance gate before answer generation; abstention accuracy was {thresholded_summary['abstention_accuracy']:.4f}.",
            f"- `hybrid_bm25_reranked` applies an OpenAI answerability reranker. The main measured gain was no-answer behavior: abstention accuracy moved from {dense_summary['abstention_accuracy']:.4f} to {reranked_summary['abstention_accuracy']:.4f}.",
            "- Chunk-level metrics are stricter than document-level metrics because finding the right document is not enough; the system must retrieve the exact supporting passage.",
            "- The takeaway is that OpenAI embeddings and BM25 are strong on clean factual questions, while reranking/answerability is the control that rejects plausible but insufficient context.",
            "",
            "## Per-Example Results",
            "",
        "| ID | Tags | Expected | Dense | Hybrid BM25 | Thresholded | Reranked |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    dense = {item["id"]: item for item in results["modes"]["dense_only"]["examples"]}
    hybrid = {item["id"]: item for item in results["modes"]["hybrid_bm25"]["examples"]}
    thresholded = {item["id"]: item for item in results["modes"]["hybrid_bm25_thresholded"]["examples"]}
    reranked = {item["id"]: item for item in results["modes"]["hybrid_bm25_reranked"]["examples"]}
    for case in cases:
        lines.append(
            f"| {case.id} | {', '.join(case.tags)} | {', '.join(case.expected_document_ids) or 'ABSTAIN'} | "
            f"{', '.join(dense[case.id]['actual_document_ids'][:3]) or 'ABSTAIN'} | "
            f"{', '.join(hybrid[case.id]['actual_document_ids'][:3]) or 'ABSTAIN'} | "
            f"{', '.join(thresholded[case.id]['actual_document_ids'][:3]) or 'ABSTAIN'} | "
            f"{', '.join(reranked[case.id]['actual_document_ids'][:3]) or 'ABSTAIN'} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "This is a controlled synthetic benchmark, not a replacement for domain-specific production evaluation. "
            "It proves the evaluation loop and exposes classes of failure, but it does not certify performance on a real company's documents. "
            "The next step is to run the same harness on real PDFs, policies, tickets, or knowledge-base exports with human-reviewed expected chunks.",
            "",
        ]
    )
    return "\n".join(lines)


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


if __name__ == "__main__":
    main()
