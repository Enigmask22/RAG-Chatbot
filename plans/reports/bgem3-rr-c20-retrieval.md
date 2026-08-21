# Retrieval eval — `bgem3-rr-c20`

- Thời điểm chạy: `2026-08-21T04:29:30+00:00`
- Số truy vấn: **242** (chấm điểm 209, bỏ qua 33 câu unanswerable)
- Config: `{"retriever": "reranked[qdrant-hybrid:rag_bgem3:rrf1-c20]:BAAI/bge-reranker-v2-m3@cuda:L512:float16:n20", "top_k": 20, "index_config": "configs\\indexing\\bgem3.yaml", "index_fingerprint": "0eaaf9265487eabb25eade5ecb6a85a74ebdbee194b1b5e215befe4bec474932", "collection": "rag_bgem3", "embedding_model": "BAAI/bge-m3", "retrieval_mode": "reranked", "branch_options": {"k": 1, "candidate_k": 20, "base": "hybrid", "rerank_candidates": 20}, "chunking": {"strategy": "hybrid", "chunk_size": 1000, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " ", ""], "min_chunk_size": 200, "max_chunk_size": 1500, "semantic_buffer_size": 1, "semantic_threshold_percentile": 85.0, "semantic_min_sentences": 3, "hybrid_max_docs_for_semantic": 5, "neighbor_context_chars": 100}, "span_resolution": {"resolved": 209, "kept_chunk_ids": 33, "unmatched_queries": [], "min_overlap_ratio": 0.5, "label_changed": 9}}`
- Môi trường: platform=Windows-11-10.0.26200-SP0, python=3.13.11

## Tổng thể

| Metric | Giá trị |
|---|---:|
| hit_rate@1 | 0.5407 |
| hit_rate@10 | 0.7129 |
| hit_rate@20 | 0.7177 |
| hit_rate@5 | 0.7033 |
| map@20 | 0.5698 |
| mrr | 0.6128 |
| ndcg@10 | 0.6075 |
| precision@1 | 0.5407 |
| precision@10 | 0.0938 |
| precision@20 | 0.0474 |
| precision@5 | 0.1809 |
| recall@1 | 0.4282 |
| recall@10 | 0.6738 |
| recall@20 | 0.6770 |
| recall@5 | 0.6555 |

## Độ trễ truy hồi (ms)

| Phân vị | ms |
|---|---:|
| mean | 231.0 |
| p50 | 232.8 |
| p95 | 263.9 |
| max | 300.0 |
| stdev | 20.5 |

## Theo nhóm truy vấn

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 34 | 0.4118 | 0.5882 | 0.5882 | 0.5588 | 0.4668 | 0.4841 | 0.4987 | 0.4118 | 0.0647 | 0.0324 | 0.1176 | 0.3824 | 0.5735 | 0.5735 | 0.5441 |
| aggregation | 26 | 0.6538 | 0.8462 | 0.8846 | 0.8462 | 0.5290 | 0.7456 | 0.6137 | 0.6538 | 0.1500 | 0.0769 | 0.2769 | 0.2949 | 0.6410 | 0.6538 | 0.5897 |
| cross_lingual | 43 | 0.3721 | 0.4651 | 0.4651 | 0.4651 | 0.4155 | 0.4147 | 0.4288 | 0.3721 | 0.0558 | 0.0279 | 0.1116 | 0.3372 | 0.4651 | 0.4651 | 0.4651 |
| factoid | 68 | 0.6176 | 0.8529 | 0.8529 | 0.8382 | 0.7080 | 0.7105 | 0.7448 | 0.6176 | 0.0868 | 0.0434 | 0.1706 | 0.6103 | 0.8529 | 0.8529 | 0.8382 |
| multi_hop | 34 | 0.6471 | 0.7941 | 0.7941 | 0.7941 | 0.6307 | 0.7083 | 0.6755 | 0.6471 | 0.1471 | 0.0750 | 0.2882 | 0.3186 | 0.7255 | 0.7353 | 0.7108 |
| table_lookup | 4 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0500 | 0.0250 | 0.1000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |

## Theo ngôn ngữ

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 82 | 0.5366 | 0.7073 | 0.7195 | 0.7073 | 0.5765 | 0.6134 | 0.6113 | 0.5366 | 0.1037 | 0.0530 | 0.2000 | 0.4065 | 0.6707 | 0.6789 | 0.6545 |
| vi | 127 | 0.5433 | 0.7165 | 0.7165 | 0.7008 | 0.5654 | 0.6124 | 0.6050 | 0.5433 | 0.0874 | 0.0437 | 0.1685 | 0.4423 | 0.6759 | 0.6759 | 0.6562 |

> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi
> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).
