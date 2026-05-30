from __future__ import annotations

from fastapi import APIRouter, Request
from sqlalchemy import text

from rag_assistant_api.domain.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=HealthResponse)
def live() -> HealthResponse:
    return HealthResponse(status="ok", checks={"api": "ok"})


@router.get("/health/ready", response_model=HealthResponse)
def ready(request: Request) -> HealthResponse:
    request.app.state.vector_store.healthcheck()
    with request.app.state.session_factory() as session:
        session.execute(text("SELECT 1"))
    return HealthResponse(status="ok", checks={"api": "ok", "database": "ok", "qdrant": "ok"})
