# Implementation Scope Notes

This repository is organized around the behaviors expected from a deployable RAG assistant:

- FastAPI API with Qdrant-backed retrieval
- File and URL ingestion
- Batch upload support
- Visible citations in the UI
- Basic retrieval metrics in responses
- Compact, dark, operational UI direction
- Offline evaluation
- Durable metadata and job storage
- Clear service/adaptor boundaries
- Docker-based local deployment

The current implementation keeps the scope intentionally single-tenant and portable. It favors clear architecture, deterministic local smoke tests, and inspectable quality metrics over enterprise-only concerns such as SSO, tenant isolation, and compliance workflows.
