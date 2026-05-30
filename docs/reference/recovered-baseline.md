# Recovered Baseline Notes

The previous local `rag-system` checkout was docs-only, but its design notes and run guide established the baseline features that this repository preserves or expands:

- FastAPI API with Qdrant-backed retrieval
- File and URL ingestion
- Batch upload support
- Visible citations in the UI
- Basic retrieval metrics in responses
- Compact, dark, operational UI direction
- Known missing pieces: evaluation pipeline, durable metadata store, and stronger architecture boundaries

This repository replaces that missing implementation with a new codebase designed for publication quality and long-term extension.
