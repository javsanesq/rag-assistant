from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rag_assistant_api.api import documents, evals, health, jobs, query
from rag_assistant_api.core.config import Settings
from rag_assistant_api.core.logging import bind_request_id
from rag_assistant_api.core.runtime import build_runtime, close_runtime

settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    if request.url.path.startswith("/api/v1") and settings.api_auth_token:
        token = request.headers.get("x-api-key")
        authorization = request.headers.get("authorization", "")
        bearer = authorization.removeprefix("Bearer ").strip() if authorization.startswith("Bearer ") else ""
        if settings.api_auth_token not in {token, bearer}:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid API credentials.", "request_id": request_id},
                headers={"x-request-id": request_id},
            )
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(evals.router)
app.include_router(jobs.router)
