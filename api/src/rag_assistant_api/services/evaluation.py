from __future__ import annotations

import json
from collections import defaultdict

from rag_assistant_api.adapters.llm import LLMProvider
from rag_assistant_api.core.config import Settings
from rag_assistant_api.domain.schemas import EvalRunRequest, QueryRequest
from rag_assistant_api.services.eval_metrics import answer_contains_expected, mean, score_ranked_ids
from rag_assistant_api.services.jobs import JobService
from rag_assistant_api.services.query import QueryService


class EvaluationService:
    def __init__(self, settings: Settings, query_service: QueryService, job_service: JobService, llm_provider: LLMProvider) -> None:
        self.settings = settings
        self.query_service = query_service
        self.job_service = job_service
        self.llm_provider = llm_provider

    def queue_run(self, request: EvalRunRequest) -> str:
        self._load_dataset(request.dataset_name)
        job = self.job_service.create_job("evaluation", request.model_dump(), dataset_name=request.dataset_name)
        return job.id

    def list_datasets(self) -> list[dict]:
        datasets = []
        for path in sorted(self.settings.effective_eval_dataset_dir.glob("*.jsonl")):
            try:
                rows = self._load_dataset(path.name)
                datasets.append({"name": path.name, "status": "valid", "examples": len(rows), "error": None})
            except Exception as exc:
                datasets.append({"name": path.name, "status": "invalid", "examples": 0, "error": str(exc)})
        return datasets

    def run_evaluation(self, job_id: str, request: EvalRunRequest) -> None:
        self.job_service.mark_running(job_id)
        try:
            dataset = self._load_dataset(request.dataset_name)
            results = []
            precision_scores = []
            hit_rates = []
            recall_scores = []
            reciprocal_ranks = []
            chunk_precision_scores = []
            chunk_hit_rates = []
            chunk_recall_scores = []
            chunk_reciprocal_ranks = []
            faithfulness_scores = []
            answer_contains_scores = []
            abstention_scores = []
            unsupported_answer_count = 0
            no_answer_count = 0
            citation_relevance_scores = []
            by_filter = defaultdict(lambda: {"examples": 0, "hits": 0})
            by_tag = defaultdict(lambda: {"examples": 0, "document_hits": 0, "chunk_hits": 0, "answer_contains": 0})
            total_examples = max(1, len(dataset))
            for index, row in enumerate(dataset, start=1):
                self.job_service.update_progress(job_id, int(((index - 1) / total_examples) * 95))
                should_answer = row.get("should_answer", True)
                query_request = QueryRequest(
                    question=row["question"],
                    document_ids=row.get("document_ids", []),
                    category=row.get("category"),
                    date_from=row.get("date_from"),
                    date_to=row.get("date_to"),
                    top_k=request.top_k or self.settings.top_k,
                    rerank=row.get("rerank", False),
                    answerability_check=row.get("answerability_check", False),
                )
                response = self.query_service.answer_question(query_request)
                top_k = request.top_k or self.settings.top_k
                expected_ids = set(row.get("expected_document_ids", []))
                expected_chunk_ids = row.get("expected_chunk_ids", [])
                actual_ids = [citation.document_id for citation in response.citations]
                actual_chunk_ids = [citation.chunk_id for citation in response.citations]
                hits = [item for item in actual_ids if item in expected_ids]
                answered = response.grounded and bool(response.used_citation_ids)
                abstained = not answered
                metric_actual_ids = [] if abstained else actual_ids
                metric_actual_chunk_ids = [] if abstained else actual_chunk_ids
                document_scores = score_ranked_ids(list(expected_ids), metric_actual_ids, top_k, should_answer)
                chunk_scores = (
                    score_ranked_ids(expected_chunk_ids, metric_actual_chunk_ids, top_k, should_answer)
                    if expected_chunk_ids
                    else document_scores
                )
                if should_answer:
                    precision = document_scores["precision_at_k"]
                    hit_rate = document_scores["hit_rate"]
                    recall = document_scores["recall_at_k"]
                    reciprocal_rank = document_scores["mrr"]
                    precision_scores.append(precision)
                    hit_rates.append(hit_rate)
                    recall_scores.append(recall)
                    reciprocal_ranks.append(reciprocal_rank)
                    chunk_precision_scores.append(chunk_scores["precision_at_k"])
                    chunk_hit_rates.append(chunk_scores["hit_rate"])
                    chunk_recall_scores.append(chunk_scores["recall_at_k"])
                    chunk_reciprocal_ranks.append(chunk_scores["mrr"])
                else:
                    precision = document_scores["precision_at_k"]
                    hit_rate = document_scores["hit_rate"]
                    recall = document_scores["recall_at_k"]
                    reciprocal_rank = document_scores["mrr"]
                    no_answer_count += 1
                    abstention_scores.append(1.0 if abstained else 0.0)
                    unsupported_answer_count += 0 if abstained else 1
                filter_key = row.get("category") or "all"
                by_filter[filter_key]["examples"] += 1
                by_filter[filter_key]["hits"] += int(hit_rate)
                used_doc_ids = {
                    citation.document_id
                    for citation in response.citations
                    if citation.chunk_id in set(response.used_citation_ids)
                }
                if should_answer:
                    citation_relevance = 1.0 if used_doc_ids & expected_ids else 0.0
                else:
                    citation_relevance = 1.0 if abstained else 0.0
                citation_relevance_scores.append(citation_relevance)
                expected_answer_contains = row.get("expected_answer_contains", [])
                answer_contains = 1.0 if answered and answer_contains_expected(response.answer, expected_answer_contains) else 0.0
                if not should_answer:
                    answer_contains = 1.0 if abstained else 0.0
                answer_contains_scores.append(answer_contains)
                tags = row.get("tags", []) or ["untagged"]
                for tag in tags:
                    by_tag[tag]["examples"] += 1
                    by_tag[tag]["document_hits"] += int(hit_rate)
                    by_tag[tag]["chunk_hits"] += int(chunk_scores["hit_rate"])
                    by_tag[tag]["answer_contains"] += int(answer_contains)
                faithfulness = self.llm_provider.judge_faithfulness(
                    row["question"],
                    response.answer,
                    [citation.model_dump(mode="json") for citation in response.citations],
                )
                faithfulness_scores.append(float(faithfulness.get("score", 0)))
                results.append(
                    {
                        "id": row.get("id"),
                        "question": row["question"],
                        "should_answer": should_answer,
                        "expected_document_ids": list(expected_ids),
                        "expected_chunk_ids": expected_chunk_ids,
                        "actual_document_ids": actual_ids,
                        "actual_chunk_ids": actual_chunk_ids,
                        "used_document_ids": sorted(used_doc_ids),
                        "used_chunk_ids": response.used_citation_ids,
                        "abstained": abstained,
                        "tags": tags,
                        "difficulty": row.get("difficulty"),
                        "answerability_reason": row.get("answerability_reason"),
                        "answer_contains": bool(answer_contains),
                        "citation_relevance": citation_relevance,
                        "precision_at_k": precision,
                        "hit_rate": hit_rate,
                        "recall_at_k": recall,
                        "mrr": reciprocal_rank,
                        "chunk_precision_at_k": chunk_scores["precision_at_k"],
                        "chunk_hit_rate": chunk_scores["hit_rate"],
                        "chunk_recall_at_k": chunk_scores["recall_at_k"],
                        "chunk_mrr": chunk_scores["mrr"],
                        "faithfulness": faithfulness,
                    }
                )
                self.job_service.update_progress(job_id, int((index / total_examples) * 95))
            summary = {
                "dataset_name": request.dataset_name,
                "examples": len(results),
                "runtime": {
                    "embed_provider": self.settings.embed_provider,
                    "embed_model": self.settings.embed_model,
                    "llm_provider": self.settings.llm_provider,
                    "llm_model": self.settings.llm_model,
                    "qdrant_vector_size": self.settings.qdrant_vector_size,
                    "top_k": request.top_k or self.settings.top_k,
                    "relevance": {
                        "min_lexical_score": self.settings.relevance_min_lexical_score,
                        "min_dense_score": self.settings.relevance_min_dense_score,
                        "min_final_score": self.settings.relevance_min_final_score,
                        "min_meaningful_terms": self.settings.relevance_min_meaningful_terms,
                    },
                },
                "precision_at_k": mean(precision_scores),
                "hit_rate": mean(hit_rates),
                "recall_at_k": mean(recall_scores),
                "mrr": mean(reciprocal_ranks),
                "chunk_precision_at_k": mean(chunk_precision_scores),
                "chunk_hit_rate": mean(chunk_hit_rates),
                "chunk_recall_at_k": mean(chunk_recall_scores),
                "chunk_mrr": mean(chunk_reciprocal_ranks),
                "answer_contains_rate": mean(answer_contains_scores),
                "faithfulness_score": mean(faithfulness_scores),
                "abstention_accuracy": mean(abstention_scores),
                "unsupported_answer_rate": round(unsupported_answer_count / max(1, no_answer_count), 4),
                "citation_relevance_rate": mean(citation_relevance_scores),
                "no_answer_examples": no_answer_count,
                "by_filter": {
                    key: {"examples": value["examples"], "hit_rate": round(value["hits"] / max(1, value["examples"]), 4)}
                    for key, value in by_filter.items()
                },
                "by_tag": {
                    key: {
                        "examples": value["examples"],
                        "document_hit_rate": round(value["document_hits"] / max(1, value["examples"]), 4),
                        "chunk_hit_rate": round(value["chunk_hits"] / max(1, value["examples"]), 4),
                        "answer_contains_rate": round(value["answer_contains"] / max(1, value["examples"]), 4),
                    }
                    for key, value in by_tag.items()
                },
            }
            self.job_service.mark_completed(job_id, {"summary": summary, "examples": results})
        except Exception as exc:  # pragma: no cover - defensive
            self.job_service.mark_failed(job_id, str(exc))

    def _load_dataset(self, dataset_name: str) -> list[dict]:
        dataset_path = self.settings.effective_eval_dataset_dir / dataset_name
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_name}")
        rows = []
        for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip():
                row = json.loads(line)
                self._validate_row(row, dataset_name, line_number)
                rows.append(row)
        return rows

    def _validate_row(self, row: dict, dataset_name: str, line_number: int) -> None:
        if not row.get("id"):
            raise ValueError(f"{dataset_name}:{line_number} is missing id.")
        if not row.get("question"):
            raise ValueError(f"{dataset_name}:{line_number} is missing question.")
        expected = row.get("expected_document_ids")
        expected_chunks = row.get("expected_chunk_ids", [])
        expected_answer_contains = row.get("expected_answer_contains", [])
        tags = row.get("tags", [])
        should_answer = row.get("should_answer", True)
        if not isinstance(expected, list):
            raise ValueError(f"{dataset_name}:{line_number} must include expected_document_ids.")
        if not isinstance(expected_chunks, list):
            raise ValueError(f"{dataset_name}:{line_number} expected_chunk_ids must be a list.")
        if isinstance(expected_answer_contains, str):
            row["expected_answer_contains"] = [expected_answer_contains]
        elif not isinstance(expected_answer_contains, list):
            raise ValueError(f"{dataset_name}:{line_number} expected_answer_contains must be a string or list.")
        if tags and not isinstance(tags, list):
            raise ValueError(f"{dataset_name}:{line_number} tags must be a list.")
        if should_answer and not expected:
            raise ValueError(f"{dataset_name}:{line_number} must include expected_document_ids when should_answer is true.")
