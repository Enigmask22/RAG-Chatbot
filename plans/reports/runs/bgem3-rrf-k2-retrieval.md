# Retrieval eval — `bgem3-rrf-k2`

- Thời điểm chạy: `2026-08-20T15:41:48+00:00`
- Số truy vấn: **242** (chấm điểm 209, bỏ qua 33 câu unanswerable)
- Config: `{"retriever": "qdrant-hybrid:rag_bgem3:rrf2-c50", "top_k": 20, "index_config": "configs\\indexing\\bgem3.yaml", "index_fingerprint": "0eaaf9265487eabb25eade5ecb6a85a74ebdbee194b1b5e215befe4bec474932", "collection": "rag_bgem3", "embedding_model": "BAAI/bge-m3", "retrieval_mode": "hybrid", "branch_options": {"k": 2}, "chunking": {"strategy": "hybrid", "chunk_size": 1000, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " ", ""], "min_chunk_size": 200, "max_chunk_size": 1500, "semantic_buffer_size": 1, "semantic_threshold_percentile": 85.0, "semantic_min_sentences": 3, "hybrid_max_docs_for_semantic": 5, "neighbor_context_chars": 100}, "span_resolution": {"resolved": 209, "kept_chunk_ids": 33, "unmatched_queries": [], "min_overlap_ratio": 0.5, "label_changed": 9}}`
- Môi trường: platform=Windows-11-10.0.26200-SP0, python=3.13.11

## Tổng thể

| Metric | Giá trị |
|---|---:|
| hit_rate@1 | 0.3301 |
| hit_rate@10 | 0.6603 |
| hit_rate@20 | 0.7129 |
| hit_rate@5 | 0.5789 |
| map@20 | 0.3955 |
| mrr | 0.4348 |
| ndcg@10 | 0.4530 |
| precision@1 | 0.3301 |
| precision@10 | 0.0847 |
| precision@20 | 0.0471 |
| precision@5 | 0.1455 |
| recall@1 | 0.2544 |
| recall@10 | 0.6037 |
| recall@20 | 0.6754 |
| recall@5 | 0.5136 |

## Độ trễ truy hồi (ms)

| Phân vị | ms |
|---|---:|
| mean | 38.1 |
| p50 | 41.9 |
| p95 | 49.6 |
| max | 58.5 |
| stdev | 9.6 |

## Theo nhóm truy vấn

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 34 | 0.2647 | 0.5000 | 0.5882 | 0.4706 | 0.3420 | 0.3491 | 0.3794 | 0.2647 | 0.0559 | 0.0324 | 0.1059 | 0.2500 | 0.5000 | 0.5735 | 0.4706 |
| aggregation | 26 | 0.3846 | 0.8846 | 0.8846 | 0.8462 | 0.4150 | 0.5849 | 0.5104 | 0.3846 | 0.1423 | 0.0769 | 0.2538 | 0.1859 | 0.6026 | 0.6603 | 0.5449 |
| cross_lingual | 43 | 0.0465 | 0.3488 | 0.4651 | 0.2326 | 0.1208 | 0.1227 | 0.1584 | 0.0465 | 0.0372 | 0.0279 | 0.0512 | 0.0465 | 0.3140 | 0.4651 | 0.2093 |
| factoid | 68 | 0.4265 | 0.8088 | 0.8529 | 0.6765 | 0.5379 | 0.5403 | 0.5995 | 0.4265 | 0.0824 | 0.0434 | 0.1382 | 0.4191 | 0.8088 | 0.8529 | 0.6765 |
| multi_hop | 34 | 0.5588 | 0.7647 | 0.7647 | 0.7353 | 0.5228 | 0.6201 | 0.5857 | 0.5588 | 0.1382 | 0.0735 | 0.2412 | 0.2745 | 0.6765 | 0.7206 | 0.5931 |
| table_lookup | 4 | 0.0000 | 0.5000 | 0.5000 | 0.5000 | 0.1750 | 0.1750 | 0.2544 | 0.0000 | 0.0500 | 0.0250 | 0.1000 | 0.0000 | 0.5000 | 0.5000 | 0.5000 |

## Theo ngôn ngữ

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 82 | 0.3415 | 0.6829 | 0.7195 | 0.5732 | 0.4101 | 0.4483 | 0.4717 | 0.3415 | 0.0988 | 0.0524 | 0.1634 | 0.2337 | 0.6260 | 0.6748 | 0.5102 |
| vi | 127 | 0.3228 | 0.6457 | 0.7087 | 0.5827 | 0.3861 | 0.4261 | 0.4409 | 0.3228 | 0.0756 | 0.0437 | 0.1339 | 0.2677 | 0.5892 | 0.6759 | 0.5157 |

> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi
> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).
