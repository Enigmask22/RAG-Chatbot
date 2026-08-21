# Retrieval eval — `chunk550`

- Thời điểm chạy: `2026-08-20T08:10:23+00:00`
- Số truy vấn: **242** (chấm điểm 209, bỏ qua 33 câu unanswerable)
- Config: `{"retriever": "qdrant-dense:rag_chunk550", "top_k": 20, "index_config": "configs\\indexing\\chunk550.yaml", "index_fingerprint": "bc6c8bbf4fad78ad824c9d1bc4203e913ebab02becb5c50e3a1d144f4827d5b3", "collection": "rag_chunk550", "embedding_model": "bkai-foundation-models/vietnamese-bi-encoder", "chunking": {"strategy": "hybrid", "chunk_size": 550, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " ", ""], "min_chunk_size": 200, "max_chunk_size": 800, "semantic_buffer_size": 1, "semantic_threshold_percentile": 85.0, "semantic_min_sentences": 3, "hybrid_max_docs_for_semantic": 5, "neighbor_context_chars": 100}, "span_resolution": {"resolved": 209, "kept_chunk_ids": 33, "unmatched_queries": [], "min_overlap_ratio": 0.5, "label_changed": 209}}`
- Môi trường: platform=Windows-11-10.0.26200-SP0, python=3.13.11

## Tổng thể

| Metric | Giá trị |
|---|---:|
| hit_rate@1 | 0.0861 |
| hit_rate@10 | 0.2584 |
| hit_rate@20 | 0.2823 |
| hit_rate@5 | 0.2010 |
| map@20 | 0.0921 |
| mrr | 0.1414 |
| ndcg@10 | 0.1215 |
| precision@1 | 0.0861 |
| precision@10 | 0.0282 |
| precision@20 | 0.0165 |
| precision@5 | 0.0431 |
| recall@1 | 0.0524 |
| recall@10 | 0.1695 |
| recall@20 | 0.1919 |
| recall@5 | 0.1295 |

## Độ trễ truy hồi (ms)

| Phân vị | ms |
|---|---:|
| mean | 26.2 |
| p50 | 30.4 |
| p95 | 31.9 |
| max | 209.1 |
| stdev | 14.0 |

## Theo nhóm truy vấn

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 34 | 0.1471 | 0.2059 | 0.2059 | 0.2059 | 0.1544 | 0.1765 | 0.1656 | 0.1471 | 0.0206 | 0.0103 | 0.0412 | 0.1324 | 0.1765 | 0.1765 | 0.1765 |
| aggregation | 26 | 0.0385 | 0.3077 | 0.3462 | 0.2308 | 0.0320 | 0.1205 | 0.0669 | 0.0385 | 0.0346 | 0.0192 | 0.0462 | 0.0077 | 0.0872 | 0.1000 | 0.0603 |
| cross_lingual | 43 | 0.0233 | 0.0233 | 0.0465 | 0.0233 | 0.0122 | 0.0245 | 0.0143 | 0.0233 | 0.0023 | 0.0023 | 0.0047 | 0.0116 | 0.0116 | 0.0233 | 0.0116 |
| factoid | 68 | 0.0735 | 0.3382 | 0.3824 | 0.2500 | 0.1329 | 0.1535 | 0.1769 | 0.0735 | 0.0338 | 0.0206 | 0.0500 | 0.0588 | 0.3088 | 0.3529 | 0.2206 |
| multi_hop | 34 | 0.1765 | 0.4412 | 0.4412 | 0.3235 | 0.1062 | 0.2626 | 0.1580 | 0.1765 | 0.0559 | 0.0324 | 0.0824 | 0.0515 | 0.1667 | 0.1912 | 0.1176 |
| table_lookup | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Theo ngôn ngữ

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 82 | 0.0488 | 0.2439 | 0.2805 | 0.2073 | 0.0551 | 0.1213 | 0.0863 | 0.0488 | 0.0268 | 0.0165 | 0.0415 | 0.0157 | 0.1252 | 0.1587 | 0.0994 |
| vi | 127 | 0.1102 | 0.2677 | 0.2835 | 0.1969 | 0.1160 | 0.1543 | 0.1442 | 0.1102 | 0.0291 | 0.0165 | 0.0441 | 0.0761 | 0.1982 | 0.2133 | 0.1490 |

> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi
> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).
