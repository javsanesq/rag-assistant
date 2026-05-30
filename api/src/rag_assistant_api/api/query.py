from __future__ import annotations

from fastapi import APIRouter, Request

from rag_assistant_api.domain.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request) -> QueryResponse:
    return request.app.state.query_service.answer_question(body)
