# Worker Notes

The v1 system uses FastAPI background tasks for ingestion and evaluation jobs so the repository stays lightweight for local Docker usage.

This directory is reserved for future extraction into a dedicated worker service. For now, the API package exposes reusable job runners that can be invoked by a CLI or a separate process later without changing the public API.
