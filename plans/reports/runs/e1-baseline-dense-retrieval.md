# Retrieval eval — `e1-baseline-dense`

- Thời điểm chạy: `2026-08-21T09:47:41+00:00`
- Số truy vấn: **242** (chấm điểm 209, bỏ qua 33 câu unanswerable)
- Config: `{"retriever": "qdrant-dense:rag_baseline", "top_k": 20, "index_config": "configs\\indexing\\baseline.yaml", "index_fingerprint": "72c87744d258ed2c068dc4869572131bd69a596a2921dff6722a09398d035d02", "collection": "rag_baseline", "embedding_model": "bkai-foundation-models/vietnamese-bi-encoder", "retrieval_mode": "dense", "branch_options": {}, "chunking": {"strategy": "hybrid", "chunk_size": 1000, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " ", ""], "min_chunk_size": 200, "max_chunk_size": 1500, "semantic_buffer_size": 1, "semantic_threshold_percentile": 85.0, "semantic_min_sentences": 3, "hybrid_max_docs_for_semantic": 5, "neighbor_context_chars": 100}, "span_resolution": {"resolved": 209, "kept_chunk_ids": 33, "unmatched_queries": [], "min_overlap_ratio": 0.5, "label_changed": 9}}`
- Môi trường: platform=Windows-11-10.0.26200-SP0, python=3.13.11

## Tổng thể

| Metric | Giá trị |
|---|---:|
| hit_rate@1 | 0.1196 |
| hit_rate@10 | 0.2775 |
| hit_rate@20 | 0.3110 |
| hit_rate@5 | 0.2153 |
| map@20 | 0.1349 |
| mrr | 0.1660 |
| ndcg@10 | 0.1621 |
| precision@1 | 0.1196 |
| precision@10 | 0.0306 |
| precision@20 | 0.0187 |
| precision@5 | 0.0459 |
| recall@1 | 0.0877 |
| recall@10 | 0.2257 |
| recall@20 | 0.2663 |
| recall@5 | 0.1746 |

## Độ trễ truy hồi (ms)

| Phân vị | ms |
|---|---:|
| mean | 25.4 |
| p50 | 22.0 |
| p95 | 42.8 |
| max | 48.1 |
| stdev | 8.6 |

## Theo nhóm truy vấn

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 34 | 0.0882 | 0.2059 | 0.2647 | 0.1765 | 0.1241 | 0.1361 | 0.1392 | 0.0882 | 0.0206 | 0.0147 | 0.0353 | 0.0735 | 0.1912 | 0.2647 | 0.1618 |
| aggregation | 26 | 0.1923 | 0.3846 | 0.4615 | 0.2308 | 0.1128 | 0.2258 | 0.1532 | 0.1923 | 0.0423 | 0.0288 | 0.0462 | 0.0897 | 0.1859 | 0.2436 | 0.1026 |
| cross_lingual | 43 | 0.0000 | 0.0233 | 0.0465 | 0.0000 | 0.0058 | 0.0058 | 0.0083 | 0.0000 | 0.0023 | 0.0023 | 0.0000 | 0.0000 | 0.0233 | 0.0465 | 0.0000 |
| factoid | 68 | 0.1471 | 0.3824 | 0.3971 | 0.3088 | 0.2152 | 0.2162 | 0.2530 | 0.1471 | 0.0382 | 0.0199 | 0.0618 | 0.1471 | 0.3750 | 0.3897 | 0.3088 |
| multi_hop | 34 | 0.2059 | 0.4118 | 0.4412 | 0.3529 | 0.1810 | 0.2720 | 0.2238 | 0.2059 | 0.0559 | 0.0353 | 0.0882 | 0.1029 | 0.2745 | 0.3480 | 0.2157 |
| table_lookup | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Theo ngôn ngữ

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 82 | 0.1220 | 0.2073 | 0.2317 | 0.1707 | 0.1075 | 0.1518 | 0.1265 | 0.1220 | 0.0220 | 0.0128 | 0.0341 | 0.0833 | 0.1463 | 0.1626 | 0.1240 |
| vi | 127 | 0.1181 | 0.3228 | 0.3622 | 0.2441 | 0.1526 | 0.1752 | 0.1851 | 0.1181 | 0.0362 | 0.0224 | 0.0535 | 0.0906 | 0.2769 | 0.3333 | 0.2073 |

> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi
> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).
