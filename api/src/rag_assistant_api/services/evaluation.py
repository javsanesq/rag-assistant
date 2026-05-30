from __future__ import annotations

import json
from collections import defaultdict

from rag_assistant_api.adapters.llm import LLMProvider
from rag_assistant_api.core.config import Settings
from rag_assistant_api.domain.schemas import EvalRunRequest, QueryRequest
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
            faithfulness_scores = []
            by_filter = defaultdict(lambda: {"examples": 0, "hits": 0})
            for row in dataset:
                query_request = QueryRequest(
                    question=row["question"],
                    document_ids=row.get("document_ids", []),
                    category=row.get("category"),
                    top_k=request.top_k or self.settings.top_k,
                )
                response = self.query_service.answer_question(query_request)
                expected_ids = set(row.get("expected_document_ids", []))
                actual_ids = [citation.document_id for citation in response.citations]
                hits = [item for item in actual_ids if item in expected_ids]
                precision = len(hits) / max(1, request.top_k or self.settings.top_k)
                hit_rate = 1.0 if hits else 0.0
                recall = len(set(hits)) / max(1, len(expected_ids))
                first_relevant_rank = next((index + 1 for index, item in enumerate(actual_ids) if item in expected_ids), None)
                reciprocal_rank = 1 / first_relevant_rank if first_relevant_rank else 0.0
                precision_scores.append(precision)
                hit_rates.append(hit_rate)
                recall_scores.append(recall)
                reciprocal_ranks.append(reciprocal_rank)
                filter_key = row.get("category") or "all"
                by_filter[filter_key]["examples"] += 1
                by_filter[filter_key]["hits"] += int(hit_rate)
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
                        "expected_document_ids": list(expected_ids),
                        "actual_document_ids": actual_ids,
                        "precision_at_k": precision,
                        "hit_rate": hit_rate,
                        "recall_at_k": recall,
                        "mrr": reciprocal_rank,
                        "faithfulness": faithfulness,
                    }
                )
            summary = {
                "dataset_name": request.dataset_name,
                "examples": len(results),
                "precision_at_k": round(sum(precision_scores) / max(1, len(precision_scores)), 4),
                "hit_rate": round(sum(hit_rates) / max(1, len(hit_rates)), 4),
                "recall_at_k": round(sum(recall_scores) / max(1, len(recall_scores)), 4),
                "mrr": round(sum(reciprocal_ranks) / max(1, len(reciprocal_ranks)), 4),
                "faithfulness_score": round(sum(faithfulness_scores) / max(1, len(faithfulness_scores)), 4),
                "by_filter": {
                    key: {"examples": value["examples"], "hit_rate": round(value["hits"] / max(1, value["examples"]), 4)}
                    for key, value in by_filter.items()
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
        if not isinstance(expected, list) or not expected:
            raise ValueError(f"{dataset_name}:{line_number} must include expected_document_ids.")
