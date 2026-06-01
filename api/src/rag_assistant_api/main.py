from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import secrets
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rag_assistant_api.api import documents, evals, health, jobs, query
from rag_assistant_api.core.config import Settings
from rag_assistant_api.core.logging import bind_request_id
from rag_assistant_api.core.runtime import build_runtime, close_runtime

settings = Settings()
logger = logging.getLogger(__name__)
_rate_limit_events: dict[tuple[str, str], list[float]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.api_auth_token:
        logger.warning("API_AUTH_TOKEN is not configured; /api/v1 routes will reject requests.")
    runtime = build_runtime()
    app.state.runtime = runtime
    for key, value in runtime.__dict__.items():
        setattr(app.state, key, value)
    yield
    close_runtime(runtime)


app = FastAPI(
    title="RAG Assistant API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.api_docs_enabled else None,
    redoc_url="/redoc" if settings.api_docs_enabled else None,
    openapi_url="/openapi.json" if settings.api_docs_enabled else None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = bind_request_id(request.headers.get("x-request-id"))
    if request.url.path.startswith("/api/v1"):
        token = request.headers.get("x-api-key")
        authorization = request.headers.get("authorization", "")
        bearer = authorization.removeprefix("Bearer ").strip() if authorization.startswith("Bearer ") else ""
        credential = token or bearer
        if not settings.api_auth_token or not _credential_matches(settings.api_auth_token, token, bearer):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid API credentials.", "request_id": request_id},
                headers={"x-request-id": request_id},
            )
        rate_limit_response = _rate_limit_response(request, credential or "anonymous", request_id)
        if rate_limit_response:
            return rate_limit_response
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


def _credential_matches(expected: str, token: str | None, bearer: str | None) -> bool:
    return any(candidate and secrets.compare_digest(expected, candidate) for candidate in (token, bearer))


def _rate_limit_response(request: Request, credential: str, request_id: str) -> JSONResponse | None:
    limit = _rate_limit_for_request(request)
    if limit <= 0:
        return None
    now = time.monotonic()
    window = settings.api_rate_limit_window_seconds
    client_host = request.client.host if request.client else "unknown"
    key = (request.url.path, f"{client_host}:{credential}")
    events = [event for event in _rate_limit_events.get(key, []) if now - event < window]
    if len(events) >= limit:
        retry_after = max(1, int(window - (now - events[0])))
        _rate_limit_events[key] = events
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded.", "request_id": request_id},
            headers={"x-request-id": request_id, "retry-after": str(retry_after)},
        )
    events.append(now)
    _rate_limit_events[key] = events
    return None


def _rate_limit_for_request(request: Request) -> int:
    if request.method != "POST":
        return 0
    if request.url.path == "/api/v1/query":
        return settings.api_query_rate_limit_per_minute
    if request.url.path == "/api/v1/documents/urls":
        return settings.api_url_ingest_rate_limit_per_minute
    return 0


app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(evals.router)
app.include_router(jobs.router)
