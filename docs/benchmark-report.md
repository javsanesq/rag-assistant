# RAG Benchmark Report

This report compares retrieval quality on a synthetic but realistic benchmark corpus using OpenAI embeddings.

## Configuration

- Embeddings: `openai` / `text-embedding-3-small`
- Vector size: `1536`
- Top K: `5`
- Reranker: `gpt-4o-mini`
- Hybrid fusion: `weighted reciprocal rank fusion` with alpha `0.65`
- Lexical backend: `SQLite FTS5 BM25 with token-overlap fallback`
- Examples: `29`

## Results

| Mode | Doc Hit | Chunk Hit | Chunk MRR | Answer Contains | Abstention Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense_only | 1.0000 | 1.0000 | 0.9800 | 0.8966 | 0.2500 |
| hybrid_bm25 | 1.0000 | 1.0000 | 0.9800 | 0.8966 | 0.2500 |
| hybrid_bm25_thresholded | 1.0000 | 1.0000 | 0.9800 | 0.8966 | 0.2500 |
| hybrid_bm25_reranked | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Results By Tag

| Mode | Tag | Examples | Doc Hit | Chunk Hit | Answer Contains | Abstention |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dense_only | compliance | 4 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| dense_only | date_conflict | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| dense_only | engineering | 5 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| dense_only | exact | 13 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| dense_only | filter | 4 | 0.7500 | 0.7500 | 0.7500 | 0.5000 |
| dense_only | lexical | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| dense_only | multi_hop | 1 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| dense_only | near_miss | 2 | 0.5000 | 0.5000 | 0.5000 | 0.0000 |
| dense_only | no_answer | 4 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| dense_only | operations | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| dense_only | policy | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| dense_only | product | 2 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| dense_only | prompt_injection | 2 | 0.5000 | 0.5000 | 0.5000 | 0.0000 |
| dense_only | sales | 2 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| dense_only | security | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| dense_only | semantic | 4 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| dense_only | support | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25 | compliance | 4 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25 | date_conflict | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_bm25 | engineering | 5 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25 | exact | 13 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25 | filter | 4 | 0.7500 | 0.7500 | 0.7500 | 0.5000 |
| hybrid_bm25 | lexical | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25 | multi_hop | 1 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25 | near_miss | 2 | 0.5000 | 0.5000 | 0.5000 | 0.0000 |
| hybrid_bm25 | no_answer | 4 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| hybrid_bm25 | operations | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25 | policy | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25 | product | 2 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25 | prompt_injection | 2 | 0.5000 | 0.5000 | 0.5000 | 0.0000 |
| hybrid_bm25 | sales | 2 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25 | security | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25 | semantic | 4 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25 | support | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25_thresholded | compliance | 4 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25_thresholded | date_conflict | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_bm25_thresholded | engineering | 5 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25_thresholded | exact | 13 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25_thresholded | filter | 4 | 0.7500 | 0.7500 | 0.7500 | 0.5000 |
| hybrid_bm25_thresholded | lexical | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25_thresholded | multi_hop | 1 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25_thresholded | near_miss | 2 | 0.5000 | 0.5000 | 0.5000 | 0.0000 |
| hybrid_bm25_thresholded | no_answer | 4 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| hybrid_bm25_thresholded | operations | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25_thresholded | policy | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25_thresholded | product | 2 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25_thresholded | prompt_injection | 2 | 0.5000 | 0.5000 | 0.5000 | 0.0000 |
| hybrid_bm25_thresholded | sales | 2 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25_thresholded | security | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25_thresholded | semantic | 4 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25_thresholded | support | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25_reranked | compliance | 4 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25_reranked | date_conflict | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_bm25_reranked | engineering | 5 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25_reranked | exact | 13 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25_reranked | filter | 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_bm25_reranked | lexical | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25_reranked | multi_hop | 1 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25_reranked | near_miss | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_bm25_reranked | no_answer | 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_bm25_reranked | operations | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25_reranked | policy | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25_reranked | product | 2 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25_reranked | prompt_injection | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid_bm25_reranked | sales | 2 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25_reranked | security | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25_reranked | semantic | 4 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid_bm25_reranked | support | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## Interpretation

- `dense_only` is the semantic-vector baseline. On this benchmark it already retrieved the expected document for every answerable example (hit rate 1.0000).
- `hybrid_bm25` adds SQLite FTS5 BM25 lexical retrieval and weighted Reciprocal Rank Fusion. On this corpus it matched dense retrieval's hit rate (1.0000), which means the answerable questions were not hard enough to expose a BM25 recall gain.
- `hybrid_bm25_thresholded` applies the production relevance gate before answer generation; abstention accuracy was 0.2500.
- `hybrid_bm25_reranked` applies an OpenAI answerability reranker. The main measured gain was no-answer behavior: abstention accuracy moved from 0.2500 to 1.0000.
- Chunk-level metrics are stricter than document-level metrics because finding the right document is not enough; the system must retrieve the exact supporting passage.
- The takeaway is that OpenAI embeddings and BM25 are strong on clean factual questions, while reranking/answerability is the control that rejects plausible but insufficient context.

## Per-Example Results

| ID | Tags | Expected | Dense | Hybrid BM25 | Thresholded | Reranked |
| --- | --- | --- | --- | --- | --- | --- |
| hr-parental-leave | exact, policy | atlas-hr-benefits | atlas-hr-benefits | atlas-hr-benefits | atlas-hr-benefits | atlas-hr-benefits |
| hr-parental-leave-semantic | semantic, policy | atlas-hr-benefits | atlas-hr-benefits | atlas-hr-benefits | atlas-hr-benefits | atlas-hr-benefits |
| hr-remote-reimbursement | exact, policy | atlas-hr-benefits | atlas-hr-benefits | atlas-hr-benefits | atlas-hr-benefits | atlas-hr-benefits |
| security-sev1-page | exact, security | atlas-security-runbook | atlas-security-runbook, atlas-prompt-injection-test | atlas-security-runbook, atlas-prompt-injection-test | atlas-security-runbook | atlas-security-runbook |
| security-keyword | lexical, security | atlas-security-runbook | atlas-security-runbook, atlas-prompt-injection-test | atlas-security-runbook, atlas-prompt-injection-test | atlas-security-runbook | atlas-security-runbook |
| prompt-injection-treatment | prompt_injection, security | atlas-prompt-injection-test | atlas-prompt-injection-test, atlas-security-runbook | atlas-prompt-injection-test, atlas-security-runbook | atlas-prompt-injection-test | atlas-prompt-injection-test |
| prompt-injection-command | prompt_injection, no_answer | ABSTAIN | atlas-prompt-injection-test, atlas-security-runbook | atlas-prompt-injection-test, atlas-security-runbook | atlas-prompt-injection-test | ABSTAIN |
| data-retention | exact, compliance | atlas-data-governance | atlas-data-governance, atlas-vendor-risk | atlas-data-governance, atlas-vendor-risk | atlas-data-governance | atlas-data-governance |
| ai-tool-policy | semantic, compliance | atlas-data-governance | atlas-data-governance, atlas-vendor-risk | atlas-data-governance, atlas-vendor-risk | atlas-data-governance, atlas-vendor-risk | atlas-data-governance |
| vendor-soc2-frequency | exact, compliance | atlas-vendor-risk | atlas-vendor-risk, atlas-data-governance | atlas-vendor-risk, atlas-data-governance | atlas-vendor-risk | atlas-vendor-risk |
| vendor-high-risk | semantic, compliance | atlas-vendor-risk | atlas-vendor-risk, atlas-data-governance | atlas-vendor-risk, atlas-data-governance | atlas-vendor-risk | atlas-vendor-risk |
| orion-panels | exact, product | atlas-product-orion | atlas-product-orion | atlas-product-orion | atlas-product-orion | atlas-product-orion |
| orion-flag | lexical, product | atlas-product-orion | atlas-product-orion | atlas-product-orion | atlas-product-orion | atlas-product-orion |
| qdrant-release | exact, engineering | atlas-release-notes-q1 | atlas-release-notes-q1, atlas-engineering-oncall | atlas-release-notes-q1, atlas-engineering-oncall | atlas-release-notes-q1 | atlas-release-notes-q1 |
| qdrant-semantic | semantic, engineering | atlas-release-notes-q1 | atlas-release-notes-q1, atlas-engineering-oncall | atlas-release-notes-q1, atlas-engineering-oncall | atlas-release-notes-q1 | atlas-release-notes-q1 |
| eval-metrics | lexical, engineering | atlas-release-notes-q1 | atlas-release-notes-q1, atlas-engineering-oncall | atlas-release-notes-q1, atlas-engineering-oncall | atlas-release-notes-q1 | atlas-release-notes-q1 |
| oncall-ack | exact, engineering | atlas-engineering-oncall | atlas-engineering-oncall, atlas-release-notes-q1 | atlas-engineering-oncall, atlas-release-notes-q1 | atlas-engineering-oncall | atlas-engineering-oncall |
| pir-draft | exact, engineering | atlas-engineering-oncall | atlas-engineering-oncall, atlas-release-notes-q1 | atlas-engineering-oncall, atlas-release-notes-q1 | atlas-engineering-oncall | atlas-engineering-oncall |
| annual-refund-current | date_conflict, filter, support | atlas-support-refunds | atlas-support-refunds | atlas-support-refunds | atlas-support-refunds | atlas-support-refunds |
| annual-refund-legacy | date_conflict, filter, support | atlas-legacy-refunds-2025 | atlas-legacy-refunds-2025 | atlas-legacy-refunds-2025 | atlas-legacy-refunds-2025 | atlas-legacy-refunds-2025 |
| monthly-refund | exact, support | atlas-support-refunds | atlas-legacy-refunds-2025, atlas-support-refunds | atlas-legacy-refunds-2025, atlas-support-refunds | atlas-legacy-refunds-2025, atlas-support-refunds | atlas-support-refunds |
| ae-discount | exact, sales | atlas-sales-discounts | atlas-sales-discounts | atlas-sales-discounts | atlas-sales-discounts | atlas-sales-discounts |
| nonprofit-discount | exact, sales | atlas-sales-discounts | atlas-sales-discounts | atlas-sales-discounts | atlas-sales-discounts | atlas-sales-discounts |
| dublin-rto-rpo | multi_hop, operations | atlas-disaster-recovery | atlas-disaster-recovery, atlas-regional-operations | atlas-disaster-recovery, atlas-regional-operations | atlas-disaster-recovery, atlas-regional-operations | atlas-disaster-recovery |
| backup-test | exact, operations | atlas-disaster-recovery | atlas-disaster-recovery, atlas-regional-operations | atlas-disaster-recovery, atlas-regional-operations | atlas-disaster-recovery | atlas-disaster-recovery |
| zurich-no-dr | near_miss, operations | atlas-regional-operations | atlas-regional-operations, atlas-disaster-recovery | atlas-regional-operations, atlas-disaster-recovery | atlas-regional-operations, atlas-disaster-recovery | atlas-regional-operations |
| filter-trap-qdrant-support | filter, no_answer | ABSTAIN | atlas-support-refunds, atlas-legacy-refunds-2025 | atlas-support-refunds, atlas-legacy-refunds-2025 | atlas-support-refunds | ABSTAIN |
| unknown-zurich-phone | near_miss, no_answer | ABSTAIN | atlas-regional-operations, atlas-disaster-recovery | atlas-regional-operations, atlas-disaster-recovery | atlas-regional-operations, atlas-disaster-recovery | ABSTAIN |
| future-refund-policy | date_conflict, filter, no_answer | ABSTAIN | ABSTAIN | ABSTAIN | ABSTAIN | ABSTAIN |

## Caveats

This is a controlled synthetic benchmark, not a replacement for domain-specific production evaluation. It proves the evaluation loop and exposes classes of failure, but it does not certify performance on a real company's documents. The next step is to run the same harness on real PDFs, policies, tickets, or knowledge-base exports with human-reviewed expected chunks.
