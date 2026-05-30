from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from rag_assistant_api.api import documents, evals, health, jobs, query
from rag_assistant_api.core.config import Settings
from rag_assistant_api.core.logging import bind_request_id
from rag_assistant_api.core.runtime import build_runtime, close_runtime


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = build_runtime()
    app.state.runtime = runtime
    for key, value in runtime.__dict__.items():
        setattr(app.state, key, value)
    yield
    close_runtime(runtime)


app = FastAPI(title="RAG Assistant API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=Settings().api_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    bind_request_id(request.headers.get("x-request-id"))
    response = await call_next(request)
    return response


app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(evals.router)
app.include_router(jobs.router)
