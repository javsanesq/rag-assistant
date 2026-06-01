# RAG Benchmark Report

This report compares retrieval quality on a synthetic but realistic benchmark corpus using OpenAI embeddings.

## Configuration

- Embeddings: `openai` / `text-embedding-3-small`
- Vector size: `1536`
- Top K: `5`
- Reranker: `gpt-4o-mini`
- Hybrid fusion: `weighted reciprocal rank fusion` with alpha `0.65`
- Lexical backend: `SQLite FTS5 BM25 with token-overlap fallback`
- Examples: `22`

## Results

| Mode | Precision@K | Hit Rate | Recall@K | MRR | Abstention Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense_only | 0.2000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 |
| hybrid_bm25 | 0.2000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 |
| hybrid_bm25_reranked | 0.2000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Interpretation

- `dense_only` is the semantic-vector baseline. On this benchmark it already retrieved the expected document for every answerable example (hit rate 1.0000).
- `hybrid_bm25` adds SQLite FTS5 BM25 lexical retrieval and weighted Reciprocal Rank Fusion. On this corpus it matched dense retrieval's hit rate (1.0000), which means the answerable questions were not hard enough to expose a BM25 recall gain.
- `hybrid_bm25_reranked` applies an OpenAI answerability reranker. The main measured gain was no-answer behavior: abstention accuracy moved from 0.5000 to 1.0000.
- Precision@K is capped at 0.2000 for this dataset because each answerable question has one expected document and the benchmark retrieves K=5 candidates.
- The takeaway is that OpenAI embeddings are already strong on clean single-hop policy questions; the reranker adds value by rejecting retrieved-but-insufficient context.

## Per-Example Results

| ID | Expected | Dense | Hybrid BM25 | Reranked |
| --- | --- | --- | --- | --- |
| hr-parental-leave | atlas-hr-benefits | atlas-hr-benefits | atlas-hr-benefits | atlas-hr-benefits |
| hr-remote-reimbursement | atlas-hr-benefits | atlas-hr-benefits | atlas-hr-benefits | atlas-hr-benefits |
| security-sev1-page | atlas-security-runbook | atlas-security-runbook | atlas-security-runbook | atlas-security-runbook |
| security-keyword | atlas-security-runbook | atlas-security-runbook | atlas-security-runbook | atlas-security-runbook |
| data-retention | atlas-data-governance | atlas-data-governance | atlas-data-governance, atlas-vendor-risk | atlas-data-governance, atlas-vendor-risk |
| ai-tool-policy | atlas-data-governance | atlas-data-governance, atlas-vendor-risk | atlas-data-governance, atlas-vendor-risk | atlas-data-governance, atlas-vendor-risk |
| orion-panels | atlas-product-orion | atlas-product-orion | atlas-product-orion | atlas-product-orion |
| orion-flag | atlas-product-orion | atlas-product-orion | atlas-product-orion | atlas-product-orion |
| qdrant-release | atlas-release-notes-q1 | atlas-release-notes-q1 | atlas-release-notes-q1 | atlas-release-notes-q1 |
| eval-metrics | atlas-release-notes-q1 | atlas-release-notes-q1 | atlas-release-notes-q1 | atlas-release-notes-q1 |
| annual-refund | atlas-support-refunds | atlas-support-refunds | atlas-support-refunds | atlas-support-refunds |
| monthly-refund | atlas-support-refunds | atlas-support-refunds | atlas-support-refunds | atlas-support-refunds |
| ae-discount | atlas-sales-discounts | atlas-sales-discounts | atlas-sales-discounts | atlas-sales-discounts |
| nonprofit-discount | atlas-sales-discounts | atlas-sales-discounts | atlas-sales-discounts | atlas-sales-discounts |
| dublin-rto | atlas-disaster-recovery | atlas-disaster-recovery | atlas-disaster-recovery | atlas-disaster-recovery |
| backup-test | atlas-disaster-recovery | atlas-disaster-recovery | atlas-disaster-recovery | atlas-disaster-recovery |
| soc2-frequency | atlas-vendor-risk | atlas-vendor-risk | atlas-vendor-risk | atlas-vendor-risk |
| vendor-high-risk | atlas-vendor-risk | atlas-vendor-risk | atlas-vendor-risk | atlas-vendor-risk |
| oncall-ack | atlas-engineering-oncall | atlas-engineering-oncall | atlas-engineering-oncall | atlas-engineering-oncall |
| pir-draft | atlas-engineering-oncall | atlas-engineering-oncall | atlas-engineering-oncall | atlas-engineering-oncall |
| filter-trap-qdrant-support | ABSTAIN | ABSTAIN | ABSTAIN | ABSTAIN |
| unknown-zurich-phone | ABSTAIN | atlas-disaster-recovery | atlas-disaster-recovery | ABSTAIN |

## Caveats

This is a controlled synthetic benchmark, not a replacement for domain-specific production evaluation. The next step is to add real company documents, expected chunk IDs, answer-quality rubrics, and prompt-injection cases.
