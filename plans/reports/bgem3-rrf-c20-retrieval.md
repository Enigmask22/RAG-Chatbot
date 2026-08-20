# Retrieval eval — `bgem3-rrf-c20`

- Thời điểm chạy: `2026-08-20T15:37:12+00:00`
- Số truy vấn: **242** (chấm điểm 209, bỏ qua 33 câu unanswerable)
- Config: `{"retriever": "qdrant-hybrid:rag_bgem3:rrf60-c20", "top_k": 20, "index_config": "configs\\indexing\\bgem3.yaml", "index_fingerprint": "0eaaf9265487eabb25eade5ecb6a85a74ebdbee194b1b5e215befe4bec474932", "collection": "rag_bgem3", "embedding_model": "BAAI/bge-m3", "retrieval_mode": "hybrid", "branch_options": {"candidate_k": 20}, "chunking": {"strategy": "hybrid", "chunk_size": 1000, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " ", ""], "min_chunk_size": 200, "max_chunk_size": 1500, "semantic_buffer_size": 1, "semantic_threshold_percentile": 85.0, "semantic_min_sentences": 3, "hybrid_max_docs_for_semantic": 5, "neighbor_context_chars": 100}, "span_resolution": {"resolved": 209, "kept_chunk_ids": 33, "unmatched_queries": [], "min_overlap_ratio": 0.5, "label_changed": 9}}`
- Môi trường: platform=Windows-11-10.0.26200-SP0, python=3.13.11

## Tổng thể

| Metric | Giá trị |
|---|---:|
| hit_rate@1 | 0.3014 |
| hit_rate@10 | 0.6364 |
| hit_rate@20 | 0.7177 |
| hit_rate@5 | 0.5359 |
| map@20 | 0.3769 |
| mrr | 0.4080 |
| ndcg@10 | 0.4313 |
| precision@1 | 0.3014 |
| precision@10 | 0.0823 |
| precision@20 | 0.0474 |
| precision@5 | 0.1349 |
| recall@1 | 0.2360 |
| recall@10 | 0.5853 |
| recall@20 | 0.6770 |
| recall@5 | 0.4753 |

## Độ trễ truy hồi (ms)

| Phân vị | ms |
|---|---:|
| mean | 33.5 |
| p50 | 32.3 |
| p95 | 47.0 |
| max | 50.9 |
| stdev | 10.3 |

## Theo nhóm truy vấn

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 34 | 0.2353 | 0.4706 | 0.5882 | 0.3824 | 0.3077 | 0.3100 | 0.3423 | 0.2353 | 0.0529 | 0.0324 | 0.0882 | 0.2353 | 0.4706 | 0.5735 | 0.3824 |
| aggregation | 26 | 0.3462 | 0.8846 | 0.8846 | 0.8077 | 0.4019 | 0.5439 | 0.5017 | 0.3462 | 0.1462 | 0.0769 | 0.2385 | 0.1731 | 0.6218 | 0.6538 | 0.5256 |
| cross_lingual | 43 | 0.0000 | 0.3256 | 0.4651 | 0.1860 | 0.0829 | 0.0814 | 0.1246 | 0.0000 | 0.0349 | 0.0279 | 0.0419 | 0.0000 | 0.3023 | 0.4651 | 0.1744 |
| factoid | 68 | 0.4118 | 0.7794 | 0.8529 | 0.6618 | 0.5324 | 0.5349 | 0.5878 | 0.4118 | 0.0794 | 0.0434 | 0.1353 | 0.4044 | 0.7794 | 0.8529 | 0.6618 |
| multi_hop | 34 | 0.5000 | 0.7353 | 0.7941 | 0.7059 | 0.4989 | 0.5760 | 0.5537 | 0.5000 | 0.1324 | 0.0750 | 0.2294 | 0.2451 | 0.6520 | 0.7353 | 0.5637 |
| table_lookup | 4 | 0.2500 | 0.5000 | 0.5000 | 0.2500 | 0.2812 | 0.2812 | 0.3289 | 0.2500 | 0.0500 | 0.0250 | 0.0500 | 0.2500 | 0.5000 | 0.5000 | 0.2500 |

## Theo ngôn ngữ

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 82 | 0.3293 | 0.6341 | 0.7195 | 0.5610 | 0.4089 | 0.4417 | 0.4570 | 0.3293 | 0.0927 | 0.0530 | 0.1585 | 0.2297 | 0.5854 | 0.6789 | 0.5020 |
| vi | 127 | 0.2835 | 0.6378 | 0.7165 | 0.5197 | 0.3562 | 0.3862 | 0.4148 | 0.2835 | 0.0756 | 0.0437 | 0.1197 | 0.2402 | 0.5853 | 0.6759 | 0.4580 |

> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi
> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).
