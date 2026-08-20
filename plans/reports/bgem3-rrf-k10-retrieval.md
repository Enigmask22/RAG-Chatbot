# Retrieval eval — `bgem3-rrf-k10`

- Thời điểm chạy: `2026-08-20T15:42:49+00:00`
- Số truy vấn: **242** (chấm điểm 209, bỏ qua 33 câu unanswerable)
- Config: `{"retriever": "qdrant-hybrid:rag_bgem3:rrf10-c50", "top_k": 20, "index_config": "configs\\indexing\\bgem3.yaml", "index_fingerprint": "0eaaf9265487eabb25eade5ecb6a85a74ebdbee194b1b5e215befe4bec474932", "collection": "rag_bgem3", "embedding_model": "BAAI/bge-m3", "retrieval_mode": "hybrid", "branch_options": {"k": 10}, "chunking": {"strategy": "hybrid", "chunk_size": 1000, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " ", ""], "min_chunk_size": 200, "max_chunk_size": 1500, "semantic_buffer_size": 1, "semantic_threshold_percentile": 85.0, "semantic_min_sentences": 3, "hybrid_max_docs_for_semantic": 5, "neighbor_context_chars": 100}, "span_resolution": {"resolved": 209, "kept_chunk_ids": 33, "unmatched_queries": [], "min_overlap_ratio": 0.5, "label_changed": 9}}`
- Môi trường: platform=Windows-11-10.0.26200-SP0, python=3.13.11

## Tổng thể

| Metric | Giá trị |
|---|---:|
| hit_rate@1 | 0.3062 |
| hit_rate@10 | 0.6411 |
| hit_rate@20 | 0.7129 |
| hit_rate@5 | 0.5120 |
| map@20 | 0.3739 |
| mrr | 0.4086 |
| ndcg@10 | 0.4305 |
| precision@1 | 0.3062 |
| precision@10 | 0.0818 |
| precision@20 | 0.0469 |
| precision@5 | 0.1292 |
| recall@1 | 0.2384 |
| recall@10 | 0.5893 |
| recall@20 | 0.6754 |
| recall@5 | 0.4514 |

## Độ trễ truy hồi (ms)

| Phân vị | ms |
|---|---:|
| mean | 34.0 |
| p50 | 28.3 |
| p95 | 49.2 |
| max | 54.5 |
| stdev | 9.4 |

## Theo nhóm truy vấn

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 34 | 0.2353 | 0.5000 | 0.5882 | 0.4118 | 0.3089 | 0.3104 | 0.3508 | 0.2353 | 0.0559 | 0.0324 | 0.0941 | 0.2353 | 0.5000 | 0.5882 | 0.4118 |
| aggregation | 26 | 0.3462 | 0.8846 | 0.8846 | 0.7692 | 0.4026 | 0.5552 | 0.4945 | 0.3462 | 0.1385 | 0.0769 | 0.2308 | 0.1731 | 0.5962 | 0.6603 | 0.5064 |
| cross_lingual | 43 | 0.0000 | 0.3023 | 0.4651 | 0.1628 | 0.0699 | 0.0706 | 0.1121 | 0.0000 | 0.0326 | 0.0267 | 0.0326 | 0.0000 | 0.2907 | 0.4535 | 0.1395 |
| factoid | 68 | 0.4118 | 0.8088 | 0.8529 | 0.6029 | 0.5250 | 0.5275 | 0.5899 | 0.4118 | 0.0824 | 0.0434 | 0.1235 | 0.4044 | 0.8088 | 0.8529 | 0.6029 |
| multi_hop | 34 | 0.5294 | 0.7353 | 0.7647 | 0.7059 | 0.5110 | 0.6008 | 0.5663 | 0.5294 | 0.1324 | 0.0735 | 0.2294 | 0.2598 | 0.6520 | 0.7206 | 0.5637 |
| table_lookup | 4 | 0.2500 | 0.2500 | 0.5000 | 0.2500 | 0.2727 | 0.2727 | 0.2500 | 0.2500 | 0.0250 | 0.0250 | 0.0500 | 0.2500 | 0.2500 | 0.5000 | 0.2500 |

## Theo ngôn ngữ

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 82 | 0.3293 | 0.6585 | 0.7317 | 0.5610 | 0.4073 | 0.4418 | 0.4617 | 0.3293 | 0.0939 | 0.0530 | 0.1610 | 0.2297 | 0.6057 | 0.6870 | 0.5061 |
| vi | 127 | 0.2913 | 0.6299 | 0.7008 | 0.4803 | 0.3522 | 0.3872 | 0.4103 | 0.2913 | 0.0740 | 0.0429 | 0.1087 | 0.2441 | 0.5787 | 0.6680 | 0.4160 |

> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi
> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).
