# Retrieval eval — `e1-rr-bgem3-reranked-ondense-rc100`

- Thời điểm chạy: `2026-08-21T09:56:20+00:00`
- Số truy vấn: **242** (chấm điểm 209, bỏ qua 33 câu unanswerable)
- Config: `{"retriever": "reranked[qdrant-dense:rag_bgem3]:BAAI/bge-reranker-v2-m3@cuda:L512:float16:n100", "top_k": 20, "index_config": "configs\\indexing\\bgem3.yaml", "index_fingerprint": "0eaaf9265487eabb25eade5ecb6a85a74ebdbee194b1b5e215befe4bec474932", "collection": "rag_bgem3", "embedding_model": "BAAI/bge-m3", "retrieval_mode": "reranked", "branch_options": {"base": "dense", "rerank_candidates": 100, "rerank_device": "cuda", "rerank_dtype": "float16"}, "chunking": {"strategy": "hybrid", "chunk_size": 1000, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " ", ""], "min_chunk_size": 200, "max_chunk_size": 1500, "semantic_buffer_size": 1, "semantic_threshold_percentile": 85.0, "semantic_min_sentences": 3, "hybrid_max_docs_for_semantic": 5, "neighbor_context_chars": 100}, "span_resolution": {"resolved": 209, "kept_chunk_ids": 33, "unmatched_queries": [], "min_overlap_ratio": 0.5, "label_changed": 9}}`
- Môi trường: platform=Windows-11-10.0.26200-SP0, python=3.13.11

## Tổng thể

| Metric | Giá trị |
|---|---:|
| hit_rate@1 | 0.5742 |
| hit_rate@10 | 0.7943 |
| hit_rate@20 | 0.7943 |
| hit_rate@5 | 0.7703 |
| map@20 | 0.6207 |
| mrr | 0.6595 |
| ndcg@10 | 0.6624 |
| precision@1 | 0.5742 |
| precision@10 | 0.1024 |
| precision@20 | 0.0522 |
| precision@5 | 0.1962 |
| recall@1 | 0.4617 |
| recall@10 | 0.7472 |
| recall@20 | 0.7568 |
| recall@5 | 0.7185 |

## Độ trễ truy hồi (ms)

| Phân vị | ms |
|---|---:|
| mean | 1040.7 |
| p50 | 1041.0 |
| p95 | 1182.3 |
| max | 1261.6 |
| stdev | 83.1 |

## Theo nhóm truy vấn

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 34 | 0.5000 | 0.7353 | 0.7353 | 0.7353 | 0.5896 | 0.6054 | 0.6262 | 0.5000 | 0.0794 | 0.0412 | 0.1529 | 0.4706 | 0.7206 | 0.7353 | 0.7059 |
| aggregation | 26 | 0.6538 | 0.9231 | 0.9231 | 0.8846 | 0.5714 | 0.7519 | 0.6423 | 0.6538 | 0.1577 | 0.0827 | 0.3077 | 0.2949 | 0.6731 | 0.7115 | 0.6538 |
| cross_lingual | 43 | 0.5116 | 0.6977 | 0.6977 | 0.6512 | 0.5704 | 0.5735 | 0.6025 | 0.5116 | 0.0791 | 0.0395 | 0.1488 | 0.4767 | 0.6977 | 0.6977 | 0.6512 |
| factoid | 68 | 0.5882 | 0.8235 | 0.8235 | 0.8088 | 0.6830 | 0.6855 | 0.7190 | 0.5882 | 0.0838 | 0.0419 | 0.1647 | 0.5809 | 0.8235 | 0.8235 | 0.8088 |
| multi_hop | 34 | 0.6471 | 0.8529 | 0.8529 | 0.8235 | 0.6425 | 0.7186 | 0.6954 | 0.6471 | 0.1559 | 0.0794 | 0.2882 | 0.3186 | 0.7696 | 0.7843 | 0.7108 |
| table_lookup | 4 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0500 | 0.0250 | 0.1000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |

## Theo ngôn ngữ

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 82 | 0.5366 | 0.7195 | 0.7195 | 0.7073 | 0.5845 | 0.6130 | 0.6175 | 0.5366 | 0.1037 | 0.0524 | 0.2024 | 0.4126 | 0.6789 | 0.6850 | 0.6667 |
| vi | 127 | 0.5984 | 0.8425 | 0.8425 | 0.8110 | 0.6440 | 0.6895 | 0.6913 | 0.5984 | 0.1016 | 0.0520 | 0.1921 | 0.4934 | 0.7913 | 0.8031 | 0.7520 |

> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi
> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).
