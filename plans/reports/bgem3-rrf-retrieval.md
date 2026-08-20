# Retrieval eval — `bgem3-rrf`

- Thời điểm chạy: `2026-08-20T15:36:41+00:00`
- Số truy vấn: **242** (chấm điểm 209, bỏ qua 33 câu unanswerable)
- Config: `{"retriever": "qdrant-hybrid:rag_bgem3:rrf60-c50", "top_k": 20, "index_config": "configs\\indexing\\bgem3.yaml", "index_fingerprint": "0eaaf9265487eabb25eade5ecb6a85a74ebdbee194b1b5e215befe4bec474932", "collection": "rag_bgem3", "embedding_model": "BAAI/bge-m3", "retrieval_mode": "hybrid", "branch_options": {}, "chunking": {"strategy": "hybrid", "chunk_size": 1000, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " ", ""], "min_chunk_size": 200, "max_chunk_size": 1500, "semantic_buffer_size": 1, "semantic_threshold_percentile": 85.0, "semantic_min_sentences": 3, "hybrid_max_docs_for_semantic": 5, "neighbor_context_chars": 100}, "span_resolution": {"resolved": 209, "kept_chunk_ids": 33, "unmatched_queries": [], "min_overlap_ratio": 0.5, "label_changed": 9}}`
- Môi trường: platform=Windows-11-10.0.26200-SP0, python=3.13.11

## Tổng thể

| Metric | Giá trị |
|---|---:|
| hit_rate@1 | 0.3014 |
| hit_rate@10 | 0.5742 |
| hit_rate@20 | 0.6746 |
| hit_rate@5 | 0.4689 |
| map@20 | 0.3583 |
| mrr | 0.3871 |
| ndcg@10 | 0.4021 |
| precision@1 | 0.3014 |
| precision@10 | 0.0742 |
| precision@20 | 0.0445 |
| precision@5 | 0.1187 |
| recall@1 | 0.2360 |
| recall@10 | 0.5327 |
| recall@20 | 0.6396 |
| recall@5 | 0.4242 |

## Độ trễ truy hồi (ms)

| Phân vị | ms |
|---|---:|
| mean | 35.8 |
| p50 | 34.9 |
| p95 | 48.6 |
| max | 52.5 |
| stdev | 9.8 |

## Theo nhóm truy vấn

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 34 | 0.2353 | 0.4706 | 0.5588 | 0.3235 | 0.2934 | 0.2937 | 0.3299 | 0.2353 | 0.0529 | 0.0309 | 0.0706 | 0.2353 | 0.4706 | 0.5588 | 0.3235 |
| aggregation | 26 | 0.3462 | 0.7308 | 0.8846 | 0.6538 | 0.3863 | 0.5170 | 0.4497 | 0.3462 | 0.1192 | 0.0769 | 0.2077 | 0.1731 | 0.5256 | 0.6603 | 0.4615 |
| cross_lingual | 43 | 0.0000 | 0.2093 | 0.3953 | 0.0698 | 0.0501 | 0.0490 | 0.0736 | 0.0000 | 0.0233 | 0.0221 | 0.0140 | 0.0000 | 0.1977 | 0.3953 | 0.0698 |
| factoid | 68 | 0.4118 | 0.7500 | 0.8235 | 0.6471 | 0.5201 | 0.5226 | 0.5716 | 0.4118 | 0.0765 | 0.0419 | 0.1324 | 0.4044 | 0.7500 | 0.8235 | 0.6471 |
| multi_hop | 34 | 0.5000 | 0.7059 | 0.7353 | 0.6471 | 0.4808 | 0.5538 | 0.5322 | 0.5000 | 0.1265 | 0.0706 | 0.2118 | 0.2451 | 0.6225 | 0.6912 | 0.5196 |
| table_lookup | 4 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.0250 | 0.0125 | 0.0500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |

## Theo ngôn ngữ

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 82 | 0.3293 | 0.6220 | 0.7073 | 0.5488 | 0.4029 | 0.4375 | 0.4529 | 0.3293 | 0.0902 | 0.0506 | 0.1561 | 0.2297 | 0.5833 | 0.6565 | 0.4898 |
| vi | 127 | 0.2835 | 0.5433 | 0.6535 | 0.4173 | 0.3296 | 0.3545 | 0.3693 | 0.2835 | 0.0638 | 0.0406 | 0.0945 | 0.2402 | 0.5000 | 0.6286 | 0.3819 |

> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi
> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).
