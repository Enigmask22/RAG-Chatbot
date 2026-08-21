# Retrieval eval — `bgem3-rr-dense-c50`

- Thời điểm chạy: `2026-08-21T04:39:31+00:00`
- Số truy vấn: **242** (chấm điểm 209, bỏ qua 33 câu unanswerable)
- Config: `{"retriever": "reranked[qdrant-dense:rag_bgem3]:BAAI/bge-reranker-v2-m3@cuda:L512:float16:n50", "top_k": 20, "index_config": "configs\\indexing\\bgem3.yaml", "index_fingerprint": "0eaaf9265487eabb25eade5ecb6a85a74ebdbee194b1b5e215befe4bec474932", "collection": "rag_bgem3", "embedding_model": "BAAI/bge-m3", "retrieval_mode": "reranked", "branch_options": {"base": "dense", "rerank_candidates": 50}, "chunking": {"strategy": "hybrid", "chunk_size": 1000, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " ", ""], "min_chunk_size": 200, "max_chunk_size": 1500, "semantic_buffer_size": 1, "semantic_threshold_percentile": 85.0, "semantic_min_sentences": 3, "hybrid_max_docs_for_semantic": 5, "neighbor_context_chars": 100}, "span_resolution": {"resolved": 209, "kept_chunk_ids": 33, "unmatched_queries": [], "min_overlap_ratio": 0.5, "label_changed": 9}}`
- Môi trường: platform=Windows-11-10.0.26200-SP0, python=3.13.11

## Tổng thể

| Metric | Giá trị |
|---|---:|
| hit_rate@1 | 0.5455 |
| hit_rate@10 | 0.7416 |
| hit_rate@20 | 0.7464 |
| hit_rate@5 | 0.7273 |
| map@20 | 0.5894 |
| mrr | 0.6265 |
| ndcg@10 | 0.6268 |
| precision@1 | 0.5455 |
| precision@10 | 0.0967 |
| precision@20 | 0.0493 |
| precision@5 | 0.1866 |
| recall@1 | 0.4354 |
| recall@10 | 0.7002 |
| recall@20 | 0.7121 |
| recall@5 | 0.6786 |

## Độ trễ truy hồi (ms)

| Phân vị | ms |
|---|---:|
| mean | 538.8 |
| p50 | 538.0 |
| p95 | 624.4 |
| max | 655.3 |
| stdev | 45.8 |

## Theo nhóm truy vấn

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 34 | 0.4706 | 0.7059 | 0.7059 | 0.7059 | 0.5628 | 0.5784 | 0.5988 | 0.4706 | 0.0765 | 0.0397 | 0.1471 | 0.4412 | 0.6912 | 0.7059 | 0.6765 |
| aggregation | 26 | 0.6538 | 0.8462 | 0.8462 | 0.8462 | 0.5736 | 0.7436 | 0.6322 | 0.6538 | 0.1500 | 0.0788 | 0.3000 | 0.2949 | 0.6410 | 0.6795 | 0.6410 |
| cross_lingual | 43 | 0.4651 | 0.6047 | 0.6279 | 0.5814 | 0.5172 | 0.5203 | 0.5393 | 0.4651 | 0.0698 | 0.0360 | 0.1349 | 0.4302 | 0.6047 | 0.6279 | 0.5814 |
| factoid | 68 | 0.5735 | 0.7941 | 0.7941 | 0.7794 | 0.6636 | 0.6636 | 0.6964 | 0.5735 | 0.0809 | 0.0404 | 0.1588 | 0.5662 | 0.7941 | 0.7941 | 0.7794 |
| multi_hop | 34 | 0.6176 | 0.8235 | 0.8235 | 0.7941 | 0.6110 | 0.6895 | 0.6662 | 0.6176 | 0.1500 | 0.0750 | 0.2765 | 0.3039 | 0.7402 | 0.7402 | 0.6814 |
| table_lookup | 4 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.0250 | 0.0125 | 0.0500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |

## Theo ngôn ngữ

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 82 | 0.5122 | 0.6707 | 0.6707 | 0.6707 | 0.5574 | 0.5823 | 0.5854 | 0.5122 | 0.0988 | 0.0500 | 0.1951 | 0.3882 | 0.6362 | 0.6423 | 0.6301 |
| vi | 127 | 0.5669 | 0.7874 | 0.7953 | 0.7638 | 0.6101 | 0.6551 | 0.6535 | 0.5669 | 0.0953 | 0.0488 | 0.1811 | 0.4659 | 0.7415 | 0.7572 | 0.7100 |

> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi
> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).
