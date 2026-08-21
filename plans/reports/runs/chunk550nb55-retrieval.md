# Retrieval eval — `chunk550nb55`

- Thời điểm chạy: `2026-08-20T08:10:47+00:00`
- Số truy vấn: **242** (chấm điểm 209, bỏ qua 33 câu unanswerable)
- Config: `{"retriever": "qdrant-dense:rag_chunk550nb55", "top_k": 20, "index_config": "configs\\indexing\\chunk550nb55.yaml", "index_fingerprint": "407b7140611fcae9469e5534482a1756139b1a8e2d17089f6db7d8d347662d3b", "collection": "rag_chunk550nb55", "embedding_model": "bkai-foundation-models/vietnamese-bi-encoder", "chunking": {"strategy": "hybrid", "chunk_size": 550, "chunk_overlap": 100, "separators": ["\n\n", "\n", ". ", " ", ""], "min_chunk_size": 200, "max_chunk_size": 800, "semantic_buffer_size": 1, "semantic_threshold_percentile": 85.0, "semantic_min_sentences": 3, "hybrid_max_docs_for_semantic": 5, "neighbor_context_chars": 55}, "span_resolution": {"resolved": 209, "kept_chunk_ids": 33, "unmatched_queries": [], "min_overlap_ratio": 0.5, "label_changed": 209}}`
- Môi trường: platform=Windows-11-10.0.26200-SP0, python=3.13.11

## Tổng thể

| Metric | Giá trị |
|---|---:|
| hit_rate@1 | 0.0861 |
| hit_rate@10 | 0.2440 |
| hit_rate@20 | 0.2919 |
| hit_rate@5 | 0.1770 |
| map@20 | 0.0953 |
| mrr | 0.1343 |
| ndcg@10 | 0.1180 |
| precision@1 | 0.0861 |
| precision@10 | 0.0268 |
| precision@20 | 0.0170 |
| precision@5 | 0.0392 |
| recall@1 | 0.0582 |
| recall@10 | 0.1591 |
| recall@20 | 0.2055 |
| recall@5 | 0.1244 |

## Độ trễ truy hồi (ms)

| Phân vị | ms |
|---|---:|
| mean | 26.9 |
| p50 | 30.4 |
| p95 | 32.2 |
| max | 205.7 |
| stdev | 13.7 |

## Theo nhóm truy vấn

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 34 | 0.1471 | 0.2059 | 0.2353 | 0.2059 | 0.1468 | 0.1627 | 0.1597 | 0.1471 | 0.0206 | 0.0118 | 0.0412 | 0.1324 | 0.1912 | 0.2059 | 0.1912 |
| aggregation | 26 | 0.0000 | 0.2308 | 0.2692 | 0.0769 | 0.0138 | 0.0506 | 0.0352 | 0.0000 | 0.0231 | 0.0154 | 0.0154 | 0.0000 | 0.0641 | 0.0846 | 0.0128 |
| cross_lingual | 43 | 0.0000 | 0.0465 | 0.0698 | 0.0465 | 0.0165 | 0.0194 | 0.0208 | 0.0000 | 0.0047 | 0.0035 | 0.0093 | 0.0000 | 0.0349 | 0.0581 | 0.0349 |
| factoid | 68 | 0.1029 | 0.3235 | 0.4118 | 0.2500 | 0.1564 | 0.1811 | 0.1849 | 0.1029 | 0.0338 | 0.0221 | 0.0529 | 0.0882 | 0.2696 | 0.3652 | 0.2108 |
| multi_hop | 34 | 0.1765 | 0.4118 | 0.4412 | 0.2647 | 0.0951 | 0.2375 | 0.1424 | 0.1765 | 0.0529 | 0.0324 | 0.0706 | 0.0490 | 0.1544 | 0.1887 | 0.0980 |
| table_lookup | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Theo ngôn ngữ

| Nhóm | n | hit_rate@1 | hit_rate@10 | hit_rate@20 | hit_rate@5 | map@20 | mrr | ndcg@10 | precision@1 | precision@10 | precision@20 | precision@5 | recall@1 | recall@10 | recall@20 | recall@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 82 | 0.0732 | 0.2317 | 0.2561 | 0.1707 | 0.0687 | 0.1261 | 0.0920 | 0.0732 | 0.0256 | 0.0165 | 0.0366 | 0.0346 | 0.1148 | 0.1565 | 0.0904 |
| vi | 127 | 0.0945 | 0.2520 | 0.3150 | 0.1811 | 0.1125 | 0.1396 | 0.1347 | 0.0945 | 0.0276 | 0.0173 | 0.0409 | 0.0735 | 0.1877 | 0.2371 | 0.1463 |

> Câu thuộc nhóm `unanswerable` không có tài liệu liên quan nên bị loại khỏi
> mọi metric xếp hạng. Chúng được đo riêng bằng refusal correctness (W5-02).
