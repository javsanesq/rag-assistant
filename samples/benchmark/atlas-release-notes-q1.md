---
title: Atlas Platform Release Notes Q1 2026
category: engineering
document_date: 2026-03-08
---

# Atlas Platform Release Notes Q1 2026

The retrieval service moved from an in-memory vector list to Qdrant-backed vector search.

The search layer now stores document metadata for category, source URI, document date, and chunk index.

The evaluation dashboard reports hit rate, precision at k, recall at k, MRR, abstention accuracy, and citation relevance.

The next planned retrieval upgrade is SQLite FTS5 BM25 lexical search followed by reranking.
